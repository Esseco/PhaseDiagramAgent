from copy import deepcopy
import json

import pytest


def task_state():
    return {
        "tasks": [
            {
                "task_id": "done",
                "task_key": "kd",
                "stage": "dft_single_point",
                "status": "completed",
                "model_version": "m",
                "upload_operation_id": "u",
            },
            {
                "task_id": "missing",
                "task_key": "km",
                "stage": "dft_single_point",
                "status": "pending",
                "model_version": "m",
                "upload_operation_id": "u",
            },
        ]
    }


def test_cancel_is_explicit_scoped_idempotent_and_not_a_dft_result():
    from phase_agent.tools.state.dft_recovery_decision import cancel_dft_recovery_waits
    from phase_agent.analysis.state.dft_round_status import dft_recovery_rounds
    from phase_agent.tools.policy.workflow_policy import active_tasks

    s = task_state()
    before = deepcopy(s)
    result = cancel_dft_recovery_waits(s, ["missing"], user_message="取消，不再回收")
    assert s == before
    assert result["tasks"][0] == before["tasks"][0]
    assert result["tasks"][1]["status"] == "cancelled"
    assert result["tasks"][1]["recovery_cancelled"] is True
    assert not active_tasks(result)
    assert cancel_dft_recovery_waits(result, ["missing"], user_message="取消，不再回收") == result
    row = dft_recovery_rounds(result)[0]
    assert row["recovered_tasks"] == 1
    assert row["resolved_tasks"] == 2
    assert row["cancelled_tasks"] == 1
    assert row["failed_tasks"] == 0
    assert not row["waived_task_ids"]


@pytest.mark.parametrize("ids", [["absent"], ["done"], ["missing", "absent"]])
def test_cancellation_never_mutates_unknown_or_completed_tasks(ids):
    from phase_agent.tools.state.dft_recovery_decision import cancel_dft_recovery_waits

    s = task_state()
    before = deepcopy(s)
    with pytest.raises(ValueError):
        cancel_dft_recovery_waits(s, ids, user_message="取消")
    assert s == before


def test_close_answer_cancels_exact_question_scope():
    from phase_agent.tools.state.dft_recovery_decision import update_dft_recovery_question

    s = update_dft_recovery_question(task_state())
    q = s["pending_dft_recovery_question"]
    result = update_dft_recovery_question(
        s, {"decision": "close", "question_id": q["question_id"], "user_message": "不再回收"}
    )
    assert result["tasks"][1]["status"] == "cancelled"
    assert result["dft_recovery_decisions"][-1]["cancelled_task_ids"] == ["missing"]
    assert not result.get("pending_dft_recovery_question")


def test_late_return_cannot_reopen_an_explicitly_cancelled_recovery():
    from phase_agent.tools.state.dft_recovery_decision import cancel_dft_recovery_waits
    from phase_agent.tools.state.reconcile_task_results import reconcile_task_results

    s = cancel_dft_recovery_waits(task_state(), ["missing"], user_message="取消")
    result = reconcile_task_results(
        s, [{**s["tasks"][1], "status": "completed", "actual_cost": 100}]
    )
    assert result["state"]["tasks"][1]["status"] == "cancelled"
    assert not result["state"].get("budget_usage")
    assert result["reconciled"][0]["reason"] == "dft_recovery_cancelled_by_user"


def legacy_training(tmp_path):
    root = tmp_path / "training"
    (root / "inputs").mkdir(parents=True)
    scope = {"data_ids": ["trained"], "cumulative_records": 1}
    plan = {
        "dataset_scope": scope,
        "original_model_version": "foundation",
        "training_round": "round-1",
    }
    (root / "inputs/training_plan.json").write_text(json.dumps(plan))
    sha = "a" * 64
    fingerprint = "b" * 64
    version = "remote-job-" + fingerprint[:12]
    return {
        "active_model_version": version,
        "active_model": {"version": version, "dataset_version": "round-1", "sha256": sha},
        "confirmed_config": {"mlip": {"version": "foundation"}},
        "model_registry": {"foundation": {"model": {"version": "foundation"}}},
        "remote_finetune_jobs": {
            "job": {
                "directory": str(root),
                "status": "results_received",
                "activated": True,
                "original_model_version": "foundation",
                "report": plan,
                "returned_results": {
                    "fingerprint": fingerprint,
                    "issues": [],
                    "models": [{"is_main_model": True, "sha256": sha}],
                },
                "training_handoff": {"candidate_model_version": version},
            }
        },
        "dft_training_records": [
            {"data_id": "trained", "structure_sha256": "g1", "checks_passed": True}
        ],
        "dft_dataset_records": [],
    }


def test_legacy_provenance_recovers_only_exact_training_identity_without_state_write(tmp_path):
    from phase_agent.analysis.convergence.training_provenance import model_with_training_provenance

    s = legacy_training(tmp_path)
    before = deepcopy(s)
    model = model_with_training_provenance(s, s["active_model"])
    assert model["training_provenance_complete"] is True
    assert model["training_data_ids"] == ["trained"]
    assert model["training_structure_hashes"] == ["g1"]
    assert s == before


@pytest.mark.parametrize(
    "bad", ["checksum", "plan_scope", "missing_geometry", "duplicate_geometry"]
)
def test_legacy_provenance_fails_closed_on_inconsistent_evidence(tmp_path, bad):
    from phase_agent.analysis.convergence.training_provenance import model_with_training_provenance

    s = legacy_training(tmp_path)
    if bad == "checksum":
        s["active_model"]["sha256"] = "c" * 64
    if bad == "plan_scope":
        s["remote_finetune_jobs"]["job"]["report"]["dataset_scope"] = {
            "data_ids": ["other"],
            "cumulative_records": 1,
        }
    if bad == "missing_geometry":
        s["dft_training_records"][0].pop("structure_sha256")
    if bad == "duplicate_geometry":
        s["dft_training_records"].append(
            {"data_id": "trained", "structure_sha256": "different", "checks_passed": True}
        )
    assert (
        model_with_training_provenance(s, s["active_model"]).get("training_provenance_complete")
        is not True
    )


def test_readiness_uses_existing_explicit_relocation_not_raw_snapshot_path(tmp_path):
    from phase_agent.runtime.project_readiness import project_readiness

    old = tmp_path / "old"
    new = tmp_path / "project"
    refs = new / "refs"
    refs.mkdir(parents=True)
    (refs / "O3.vasp").write_text("fixture")
    cfg = {"system": {"phase_references": {"O3": str(old / "refs/O3.vasp")}}}
    settings = {"local_path_relocations": [{"from": str(old), "to": str(new)}]}
    before = deepcopy(cfg)
    report = project_readiness({}, cfg, settings, workspace=new)
    assert report["references"]["missing_phases"] == []
    assert cfg == before


def test_independent_validation_uses_restored_geometry_exclusions(tmp_path):
    from phase_agent.analysis.convergence.independent_evidence import independent_records

    s = legacy_training(tmp_path)
    version = s["active_model_version"]
    s["dft_dataset_records"] = [
        {
            "task_id": "repeat",
            "data_id": "different-id",
            "structure_sha256": "g1",
            "model_version": version,
            "stage": "dft_single_point",
        },
        {
            "task_id": "new",
            "data_id": "new-id",
            "structure_sha256": "g2",
            "model_version": version,
            "stage": "dft_single_point",
        },
    ]
    result = independent_records(s, version)
    assert [r["task_id"] for r in result["records"]] == ["new"]
    assert result["excluded"] == ["repeat"]


def test_cancelled_dft_closes_recovery_without_counting_as_validated():
    from phase_agent.tools.state.dft_recovery_decision import cancel_dft_recovery_waits
    from phase_agent.analysis.state.post_dft_assessment import post_dft_assessment
    from tests.test_dft_comparison_csv import add_result

    s = {}
    add_result(s)
    s["tasks"].append({**s["tasks"][0], "task_id": "missing", "status": "pending"})
    s = cancel_dft_recovery_waits(s, ["missing"], user_message="不再回收")
    report = post_dft_assessment(s)
    assert report["status"] == "evaluated"
    assert report["recovered_tasks"] == 1
    assert report["cancelled_tasks"] == 1
    assert report["eligible_structures"] == 1
    assert report["failed_tasks"] == 0


def test_cancellation_reason_is_available_in_bounded_agent_context():
    from phase_agent.tools.state.dft_recovery_decision import cancel_dft_recovery_waits
    from phase_agent.analysis.state.build_decision_context import build_decision_context

    s = cancel_dft_recovery_waits(task_state(), ["missing"], user_message="不再回收")
    facts = build_decision_context(s)["dft_recovery_cancellations"]
    assert facts[0]["task_id"] == "missing"
    assert facts[0]["user_message"] == "不再回收"
    assert facts[0]["remote_cancel_sent"] is False

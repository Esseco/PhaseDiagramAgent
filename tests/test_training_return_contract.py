from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import pytest

from phase_agent.graphs.dialogue.saved_control import semantic_recovery_intent
from phase_agent.tools.workflows.tool_outcomes import _apply_execution_result


@pytest.mark.parametrize("message", ["已回收", "已回传", "回收完成。", "回收完毕", "结果已回传"])
def test_return_notice_starts_observation_without_intent_model(message):
    handler = SimpleNamespace(
        workflow_kwargs={"agent_client": lambda _: pytest.fail("no intent model")}
    )
    assert semantic_recovery_intent(handler, {}, message) == "approve"


def delivered():
    handoff = {"folder": "input", "import_directory": "job", "package_fingerprint": "bound"}
    action = {
        "tool": "update_mlip",
        "task_key": "prepare",
        "budget": 0,
        "parameters": {"prepare_inputs_only": True},
    }
    state = {
        "active_model_version": "old",
        "confirmed_config_version": "c",
        "remote_finetune_jobs": {
            "j": {
                "directory": "job",
                "handoff": handoff,
                "original_model_version": "old",
                "status": "results_received",
            }
        },
        "execution_recovery_report": {"status": "clear", "unsettled": []},
        "budget_remaining": 17,
        "budget_reservations": {
            "prepare": {
                "status": "reserved",
                "stage": "update_mlip",
                "reserved_cost": 0,
                "relative_cost": 0,
                "config_version": "c",
            }
        },
        "effective_decisions": {"prepare": {"status": "reserved", "tool": "update_mlip"}},
    }
    execution = {
        "status": "completed",
        "result": {"status": "awaiting_remote_training", "handoffs": [handoff], "submitted": False},
    }
    return state, action, execution


def test_successful_input_delivery_settles_only_the_local_preparation():
    state, action, execution = delivered()
    updated, status = _apply_execution_result(state, action, execution, record_id="r", formal=True)
    assert status == "awaiting_remote_training"
    assert updated["budget_reservations"]["prepare"]["status"] == "settled"
    assert updated["effective_decisions"]["prepare"]["status"] == "completed"
    assert updated["remote_finetune_jobs"]["j"]["status"] == "results_received"
    assert updated["active_model_version"] == "old" and updated["budget_remaining"] == 17


def test_saved_delivery_repair_unblocks_old_checkpoint_idempotently():
    from phase_agent.tools.workflows.training_input_settlement import settle_saved_training_inputs
    from phase_agent.graphs.project.runner import (
        _business_state_is_clean_for_stale_checkpoint as clean,
    )

    state, action, execution = delivered()
    state["action_records"] = [
        {"record_id": "r", "final_action": action, "execution_result": execution}
    ]
    assert not clean("state", state=state)
    updated = settle_saved_training_inputs(state)
    assert clean("state", state=updated)
    assert settle_saved_training_inputs(updated) == updated


@pytest.mark.parametrize("bad", ["unbound", "cost", "unknown", "reconciliation"])
def test_input_settlement_keeps_unproven_or_scientific_reservations(bad):
    from phase_agent.tools.workflows.training_input_settlement import settle_saved_training_inputs

    state, action, execution = delivered()
    state["action_records"] = [
        {"record_id": "r", "final_action": action, "execution_result": execution}
    ]
    if bad == "unbound":
        execution["result"]["handoffs"] = [{"folder": "other", "import_directory": "job"}]
    elif bad == "cost":
        state["budget_reservations"]["prepare"]["reserved_cost"] = 1
    elif bad == "unknown":
        execution["status"] = "unknown"
    else:
        state["budget_reservations"]["prepare"]["reconciliation_required"] = True
    before = deepcopy(state)
    assert settle_saved_training_inputs(state) == before


def test_acknowledged_training_return_still_enters_analysis(tmp_path, monkeypatch):
    from phase_agent.graphs.dialogue.nodes import read_project
    import phase_agent.tools.workflows.recover_saved_results as recovery
    from phase_agent.tools.step_runner.file_protocol import write_json

    state, action, execution = delivered()
    state["action_records"] = [
        {"record_id": "r", "final_action": action, "execution_result": execution}
    ]
    state["remote_finetune_jobs"]["j"]["returned_results"] = {"status": "validation_required"}
    path = tmp_path / "state.json"
    write_json(path, state)
    monkeypatch.setattr(
        recovery, "recovery_candidate", lambda *a, **kw: {"pending": False, "fingerprint": "f"}
    )
    monkeypatch.setattr(recovery, "recover_saved_results", lambda s, **kw: s)
    handler = SimpleNamespace(
        state_path=path, conversation_id=None, config_delegate=None, workflow_kwargs={}
    )
    runtime = SimpleNamespace(
        context=SimpleNamespace(
            handler=handler, conversation_id="t", messages=[{"role": "user", "content": "已回收"}]
        )
    )
    result = read_project({}, runtime)
    assert result["operation"] == "dispatch" and result["recovery_just_approved"]
    # The shared scientific runner settles delivery after dialogue dispatch.
    assert result["facts"]["budget_reservations"]["prepare"]["status"] == "reserved"


def test_training_markdown_reports_all_cv_and_matched_baseline_separately(tmp_path):
    from tests.test_remote_training_recovery import returned_job
    from phase_agent.tools.local.training_cv_review import prepare_cv_review

    job, results = returned_job(tmp_path)
    job["report"]["epoch"] = "epoch1"
    report, message = prepare_cv_review({}, job, {"fingerprint": "test"})
    path = Path(report["report_path"])
    assert path.is_file() and path.suffix == ".md" and str(path) in message
    text = path.read_text(encoding="utf-8")
    assert "epoch1" in text and "epoch2" in text
    assert "折外" in text and "独立测试" in text and "匹配" in text
    assert report["structures"] == 0 and report["activatable"] is False
    assert "缺少匹配数据" in text
    stamp = path.stat().st_mtime_ns
    prepare_cv_review({}, job, {"fingerprint": "test"})
    assert path.stat().st_mtime_ns == stamp


def test_report_and_epoch_approval_survive_short_chat_reply():
    from phase_agent.runtime.training_reply import concise_training_reply
    from phase_agent.runtime.interaction_reports import short_reply

    handoff = {
        "stage": "awaiting_activation_approval",
        "candidate_model_version": "new",
        "training_report_path": "E:/project/results/finetuning_results.md",
        "source_epoch": "epoch1",
        "next_epoch": "epoch2",
    }
    state = {
        "active_model_version": "old",
        "candidate_models": {
            "new": {"agent_review": {"choice": "activate", "reason": "训练结果已核对"}}
        },
    }
    reply = short_reply(concise_training_reply(handoff, state))
    assert "finetuning_results.md" in reply and "epoch2" in reply
    assert "是否" in reply and "同意" in reply and "拒绝" in reply
    assert state["active_model_version"] == "old"


def training_analysis_case(tmp_path):
    import csv
    import json
    from tests.test_remote_training_recovery import returned_job
    from phase_agent.tools.local.training_handoff import advance_training_handoffs
    from phase_agent.tools.local.training_agent_review import review_training_choice
    from phase_agent.runtime.review_requests import pending_reviews
    from phase_agent.runtime.workflow_reply_presentation import format_workflow_reply
    from phase_agent.runtime.interaction_reports import short_reply

    job, root = returned_job(tmp_path)
    job["report"]["epoch"] = "epoch1"
    rows = {
        "energy_comparison.csv": [
            {"task_id": "t", "fold": "fold_1", "E_DFT_eV_per_atom": 1, "E_MLIP_eV_per_atom": 1.001}
        ],
        "force_comparison.csv": [
            {
                "task_id": "t",
                "fold": "fold_1",
                "atom_index": 0,
                "component": i,
                "F_DFT_eV_per_A": 1,
                "F_MLIP_eV_per_A": 1.002,
            }
            for i in range(3)
        ],
    }
    for name, values in rows.items():
        with (root / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    data = list(csv.DictReader((root / "kfold_metrics.csv").open()))
    for row in data:
        row["force_components"] = 3
    with (root / "kfold_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    record = {
        "task_id": "t",
        "model_version": "base",
        "status": "completed",
        "converged": True,
        "checks_passed": True,
        "training_ready": True,
        "atom_count": 1,
        "composition": {"Cr": 1},
        "energy": 1,
        "energy_unit": "eV",
        "forces": [[1, 1, 1]],
        "forces_unit": "eV/angstrom",
        "mlip_prediction": {
            "model_version": "base",
            "energy": 1.01,
            "energy_unit": "eV",
            "forces": [[1.02, 1.02, 1.02]],
            "forces_unit": "eV/angstrom",
        },
    }
    state = {
        "active_model_version": "base",
        "remote_finetune_jobs": {"j": job},
        "dft_dataset_records": [record],
        "dft_mlip_comparisons": [{"task_id": "t", "status": "completed"}],
    }
    return state


def test_real_training_analysis_creates_report_and_saved_epoch_review(tmp_path):
    import json
    from phase_agent.tools.local.training_handoff import advance_training_handoffs
    from phase_agent.tools.local.training_agent_review import review_training_choice
    from phase_agent.runtime.review_requests import pending_reviews
    from phase_agent.runtime.workflow_reply_presentation import format_workflow_reply
    from phase_agent.runtime.interaction_reports import short_reply

    state = training_analysis_case(tmp_path)
    current, waits = advance_training_handoffs(state, {})
    current, waits = review_training_choice(
        current, waits, lambda _: pytest.fail("no decision model")
    )
    assert len(waits) == 1 and waits[0]["stage"] == "awaiting_activation_approval"
    assert current["active_model_version"] == "base"
    reviews = pending_reviews(current)
    assert len(reviews) == 1 and reviews[0]["review_kind"] == "model_activation"
    text = short_reply(
        format_workflow_reply(
            {
                "status": "training_handoff",
                "state": current,
                "training_handoffs": waits,
                "reason": waits[0]["reason"],
            },
            tmp_path / "state.json",
        )
    )
    assert "finetuning_results.md" in text and "是否进入 epoch2" in text
    again, again_waits = advance_training_handoffs(json.loads(json.dumps(current)), {})
    again, _ = review_training_choice(again, again_waits, None)
    assert again["active_model_version"] == "base"
    assert pending_reviews(again)[0]["proposal_hash"] == reviews[0]["proposal_hash"]


def test_partial_baseline_is_disclosed_in_manual_epoch_review(tmp_path):
    from tests.test_remote_training_recovery import returned_job
    from phase_agent.tools.local.training_cv_review import register_cv_candidate
    from phase_agent.tools.local.recover_remote_training import inspect_training_results

    job, root = returned_job(tmp_path)
    result = inspect_training_results(job)
    report = {
        "structures": 1,
        "returned_structures": 2,
        "force_components": 1,
        "old": {
            "energy_per_atom": {"mae": 0.01, "rmse": 0.02},
            "forces": {"mae": 0.1, "rmse": 0.2},
        },
        "new": {
            "energy_per_atom": {"mae": 0.001, "rmse": 0.002},
            "forces": {"mae": 0.01, "rmse": 0.02},
        },
        "limitations": [],
        "report_path": str(root / "finetuning_results.md"),
    }
    (root / "cv_baseline_review.json").write_text("review")
    state = {"active_model_version": "base"}
    handoff = {}
    assert register_cv_candidate(state, "j", job, result, report, handoff)
    candidate = state["candidate_models"][handoff["candidate_model_version"]]
    assert candidate["validation"]["checks"]["fully_matched"] is False
    assert candidate["validation"]["energy_and_force_improved"] is False
    assert "1/2" in " ".join(candidate["validation"]["limitations"])
    assert state["active_model_version"] == "base"

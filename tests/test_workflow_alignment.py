"""Confirmed scientific contract, tested only on in-memory/fake project facts."""

from copy import deepcopy

import pytest

from tests.test_single_point_convergence import state as paired_state
from phase_agent.configuration.defaults.default_layered_search_config import (
    default_layered_search_config,
)


def evidence():
    s = paired_state()
    s["active_model"] = {"version": "m1", "training_data_ids": [], "training_structure_hashes": []}
    for r in s["dft_dataset_records"]:
        r["data_id"] = "data-" + r["task_id"]
        r["structure_sha256"] = "geometry-" + r["structure_id"]
        r["checks_passed"] = True
    for r in s["phase_diagrams"]["mlip"]["entries"]:
        r.update(is_stable=True, ehull=0.0, structure_sha256="geometry-" + r["structure_id"])
    s["feedback_cycles"] = [
        {
            "cycle_id": "c1",
            "model_version": "m1",
            "method": "mlip",
            "complete": True,
            "hull_change": 0.001,
            "ground_state_unchanged": True,
        }
    ]
    return s


def test_new_defaults_are_independent_mlip_feedback_contract():
    c = default_layered_search_config()["convergence"]
    assert c["stable_feedback_cycles"] == 1
    assert c["final_energy_mae_tolerance"] == 0.005
    assert c["final_force_rmse_tolerance"] == 0.05
    assert c["minimum_ground_state_dft_fraction"] == 0.6


def test_current_untrained_dft_passes_but_training_geometry_is_excluded():
    from phase_agent.analysis.convergence.independent_evidence import independent_records

    s = evidence()
    assert len(independent_records(s, "m1")["records"]) == 2
    s["active_model"]["training_structure_hashes"] = ["geometry-a"]
    assert [r["task_id"] for r in independent_records(s, "m1")["records"]] == ["b"]
    s["active_model"]["training_data_ids"] = ["data-b"]
    assert not independent_records(s, "m1")["records"]


def test_trained_model_without_training_provenance_is_not_independent():
    from phase_agent.analysis.convergence.independent_evidence import independent_records

    s = evidence()
    s["active_model"] = {"version": "m1", "dataset_version": "trained-unknown"}
    assert independent_records(s, "m1")["status"] == "missing_training_provenance"


def test_feedback_contract_requires_cycles_and_exact_ground_state_geometry():
    from phase_agent.analysis.convergence.check_global_convergence import check_global_convergence

    s = evidence()
    rules = default_layered_search_config()["convergence"]
    assert check_global_convergence(s, rules=rules)["criteria_satisfied"]
    s["dft_dataset_records"][0]["structure_sha256"] = "different-geometry"
    result = check_global_convergence(s, rules=rules)
    assert not result["checks"]["ground_state_dft_validated"]
    s = evidence()
    s.pop("feedback_cycles")
    assert not check_global_convergence(s, rules=rules)["criteria_satisfied"]


@pytest.mark.parametrize("field,value", [("energy", 0.005), ("forces", 0.05)])
def test_exact_delivery_error_limit_does_not_pass(field, value):
    from phase_agent.analysis.convergence.check_global_convergence import check_global_convergence

    s = evidence()
    for r in s["dft_dataset_records"]:
        if field == "energy":
            r["mlip_prediction"]["energy"] = r["energy"] + value
        else:
            r["mlip_prediction"]["forces"] = [[x + value for x in r["forces"][0]]]
    result = check_global_convergence(s, rules=default_layered_search_config()["convergence"])
    assert not result["criteria_satisfied"]


def test_feedback_recorder_is_mlip_only_and_idempotent():
    from phase_agent.analysis.convergence.feedback_cycles import record_feedback_cycle

    s = evidence()
    s.pop("feedback_cycles")
    s["tasks"].append(
        {"task_id": "mc-1", "stage": "deep_search", "status": "completed", "model_version": "m1"}
    )
    s["phase_diagrams"]["dft"] = {
        "entries": [{"structure_id": "d", "ehull": 123, "is_stable": True}]
    }
    for r in s["dft_dataset_records"]:
        s["tasks"].append(
            {
                "task_id": r["task_id"],
                "stage": "dft_single_point",
                "status": "completed",
                "model_version": "m1",
                "batch_id": "batch",
                "structure_id": r["structure_id"],
                "source_search_task_id": "mc-1",
            }
        )
    record_feedback_cycle(s)
    assert s["feedback_cycles"][-1]["method"] == "mlip"
    record_feedback_cycle(s)
    assert len(s["feedback_cycles"]) == 1
    s["tasks"].append(
        {
            "task_id": "waiting",
            "stage": "dft_single_point",
            "status": "running",
            "model_version": "m1",
            "batch_id": "next",
            "recovery_wait_waived": True,
        }
    )
    record_feedback_cycle(s)
    assert len(s["feedback_cycles"]) == 1


def test_training_barrier_does_not_waive_actually_running_dft():
    from phase_agent.tools.policy.workflow_policy import active_tasks

    assert active_tasks(
        {
            "tasks": [
                {
                    "task_id": "x",
                    "status": "running",
                    "stage": "dft_single_point",
                    "recovery_wait_waived": True,
                }
            ]
        }
    )


def test_shared_data_disabled_by_default_and_explicit_sources_required(tmp_path):
    from phase_agent.persistence.shared_dft import select

    s = {"shared_dft": {"directory": str(tmp_path / "absent"), "diagram_id": "here"}}
    assert select(s, {}) == []
    assert not (tmp_path / "absent").exists()


def test_advisory_dft_recommendation_does_not_discard_on_cost():
    from phase_agent.decisions.qbc_selection.recommend_post_mc_dft import recommend_post_mc_dft

    config = default_layered_search_config()
    config["budgets"]["total_relative_cost"] = 0
    pool = [
        {
            "candidate_id": str(i),
            "phase": "O3",
            "x_Na_per_O2": i / 10,
            "atom_count": 4,
            "predicted_Ehull": 0.0,
        }
        for i in range(10)
    ]
    assert recommend_post_mc_dft(pool, config)["summary"]["selected"] > 0


def test_small_dft_batch_keeps_independent_audit():
    from phase_agent.decisions.qbc_selection.select_dft_candidates import select_dft_candidates

    pool = [{"candidate_id": str(i), "hull_impact": i} for i in range(10)]
    result = select_dft_candidates(pool, batch_size=2, seed=0, audit_fraction=0.05)
    assert result["summary"]["audit_selected"] >= 1


def test_new_proposal_uses_existing_100_to_1000_contract(monkeypatch):
    import phase_agent.decisions.agent.propose_tool_action as module

    captured = []

    def fake(client, payload):
        captured.append(payload)
        raise ValueError("test stops before scientific action")

    monkeypatch.setattr(module, "request_validated_action", fake)
    module.propose_agent_tool_action(
        {}, agent_client=lambda x: {}, allowed_tools=["generate_branches"], config={}
    )
    p = captured[0]["decision_context"]["branch_count_policy"]
    assert p["minimum"] == 100 and p["maximum"] == 1000


def test_training_complete_does_not_request_another_scientific_choice():
    from phase_agent.tools.local.training_agent_review import review_training_choice

    handoff = {
        "stage": "awaiting_activation_approval",
        "candidate_model_version": "new",
        "evidence_type": "grouped_cross_validation_review",
    }
    s = {
        "active_model_version": "old",
        "candidate_models": {"new": {"validation": {"status": "completed", "passed": True}}},
    }
    calls = []
    updated, waits = review_training_choice(s, [handoff], lambda p: calls.append(p))
    assert calls == []
    assert waits[0]["direction_proposal"]["direction"] == "activate"
    assert updated["active_model_version"] == "old"


def test_invalid_dft_review_fails_without_format_repair_call():
    from phase_agent.decisions.agent.post_dft_review import request_review_repair

    calls = []
    with pytest.raises(ValueError):
        request_review_repair(
            lambda p: calls.append(p),
            {"allowed_tools": ["update_mlip"], "instruction": ""},
            {"tool": "update_mlip", "parameters": {}},
        )
    assert calls == []


def test_shared_snapshot_uses_named_sources_and_is_immutable(tmp_path, monkeypatch):
    from phase_agent.persistence import shared_dft as shared

    config = {
        "shared_dft": {
            "enabled": True,
            "directory": str(tmp_path),
            "diagram_id": "here",
            "source_projects": ["A"],
        },
        "system": {"boundary": {"TM_ratio": {"Ni": 1}}},
        "dft": {},
    }
    monkeypatch.setattr(shared, "in_boundary", lambda p, c: True)

    def insert(key, source):
        import json

        with shared.database(config["shared_dft"]) as db:
            db.execute(
                "INSERT INTO records VALUES (?,?,?,?,?,?,?)",
                (
                    key,
                    shared.identity(config),
                    shared.protocol(config),
                    "dft_single_point",
                    None,
                    "p",
                    json.dumps({"source_diagram": source}),
                ),
            )

    insert("one", "A")
    insert("other", "B")
    snapshot = shared.capture_snapshot({}, config)
    assert snapshot["record_ids"] == ["one"]
    insert("two", "A")
    s = {"shared_dft_snapshot": snapshot}
    assert [key for key, p in shared.select(s, config)] == ["one"]
    assert shared.capture_snapshot(s, config)["record_ids"] == ["one", "two"]


def test_supercell_revision_waits_for_actual_task_end_without_rewriting_tasks():
    from phase_agent.configuration.runtime.authorize_budget_extension import (
        authorize_budget_extension,
    )

    old = {"system": {"boundary": {"H": [1]}}, "budgets": {"structure_limits": {"max_det_H": 1}}}
    new = deepcopy(old)
    new["system"]["boundary"]["H"] = [1, 2]
    new["budgets"]["structure_limits"]["max_det_H"] = 2
    task = {
        "task_id": "t",
        "status": "running",
        "config_version": "old",
        "recovery_wait_waived": True,
    }
    s = {"confirmed_config_version": "old", "confirmed_config": old, "tasks": [task]}
    snapshot = {"config_version": "new", "config": new}
    result = authorize_budget_extension(s, snapshot)
    assert result["status"] == "deferred_supercell_revision"
    assert result["state"]["confirmed_config_version"] == "old"
    assert result["state"]["tasks"] == [task]
    s["tasks"][0]["status"] = "completed"
    result = authorize_budget_extension(s, snapshot)
    assert result["status"] == "supercell_revision_applied"
    assert result["state"]["confirmed_config_version"] == "new"
    assert result["state"]["tasks"][0]["config_version"] == "old"


def test_dft_relax_needs_objective_geometry_and_hull_evidence():
    from phase_agent.tools.workflows.validate_dft_agent_decisions import (
        validate_dft_agent_decisions,
    )

    config = {"allowed_actions": ["DFT_RELAX"], "action_costs": {"DFT_RELAX": 1}}
    proposal = {"decisions": [{"candidate_id": "a", "action": "DFT_RELAX", "reason": "near hull"}]}
    metric = {"candidate_id": "a", "predicted_Ehull": 0.001, "atom_count": 1}
    result = validate_dft_agent_decisions(
        proposal, [metric], {}, config=config, config_version="c", remaining_budget=100
    )
    assert not result["accepted"]
    metric["geometry_optimization_evidence"] = {
        "source_task_id": "dft-a",
        "residual_force_ev_per_angstrom": 0.2,
        "force_tolerance_ev_per_angstrom": 0.05,
    }
    result = validate_dft_agent_decisions(
        proposal, [metric], {}, config=config, config_version="c", remaining_budget=100
    )
    assert result["accepted"]


def test_activation_cannot_bypass_real_inflight_task_barrier():
    from phase_agent.persistence.models.activate_validated_model import activate_validated_model

    s = {
        "active_model_version": "old",
        "tasks": [{"status": "running", "stage": "dft_single_point", "recovery_wait_waived": True}],
    }
    result = activate_validated_model(
        s,
        {"version": "new"},
        {"status": "completed", "passed": True},
        approval_reason="approved switch",
    )
    assert result["status"] != "activated"
    assert result["state"]["active_model_version"] == "old"


def test_first_round_normal_feedback_requires_initial_finetune():
    from phase_agent.tools.policy.workflow_policy import initial_direction_errors

    assessment = {"status": "evaluated", "eligible_structures": 10, "minimum_new_dft_records": 10}
    assert initial_direction_errors({"tool": "generate_branches"}, {}, assessment)
    assert not initial_direction_errors({"tool": "update_mlip"}, {}, assessment)
    assert not initial_direction_errors(
        {"tool": "generate_branches"},
        {"active_model": {"dataset_version": "already-trained"}},
        assessment,
    )


def test_startup_approval_does_not_recapture_new_reference_records(monkeypatch):
    from phase_agent.tools.workflows.reference_dft_startup import startup_gate
    from phase_agent.persistence import shared_dft

    frozen = {"source_projects": ["A"], "record_ids": ["old"], "fingerprint": "saved"}
    state = {
        "reference_dft_startup": {
            "status": "pending",
            "references": [{"record_id": "old"}],
            "shared_dft_snapshot": frozen,
        }
    }
    seen = []
    monkeypatch.setattr(
        shared_dft, "capture_snapshot", lambda *a: pytest.fail("must not recapture")
    )
    monkeypatch.setattr(shared_dft, "accept_snapshot", lambda s, c, v: seen.append(v))
    assert startup_gate(state, {}, message="foundation") is None
    assert seen == [frozen]


def test_custom_training_contract_waits_for_activation_even_without_metric_gain(tmp_path):
    from phase_agent.tools.workflows.create_model_update_handler import create_model_update_handler

    path = tmp_path / "model"
    path.write_bytes(b"model")
    handler = create_model_update_handler(trainer=lambda **kw: {"model_path": str(path)})
    state = {
        "active_model_version": "old",
        "active_model": {"version": "old"},
        "new_dft_records": [
            {"data_id": "d", "status": "completed", "converged": True, "energy": 1.0}
        ],
    }
    result = handler(
        trigger={},
        state=state,
        manager=None,
        config={
            **default_layered_search_config(),
            "mlip_finetune": {
                "minimum_new_dft_records": 1,
                "training": {"minimum_new_dft_records": 1},
            },
        },
    )
    assert result["status"] == "awaiting_activation_approval"
    assert result["state"]["active_model_version"] == "old"
    assert result["validation"]["independent_test_completed"] is False


def test_historical_dft_cannot_complete_an_unrelated_new_search_cycle():
    from phase_agent.analysis.convergence.feedback_cycles import record_feedback_cycle

    s = evidence()
    s.pop("feedback_cycles")
    s["tasks"] = [
        {
            "task_id": "old-dft",
            "stage": "dft_single_point",
            "status": "completed",
            "model_version": "m1",
            "search_group_index": 1,
            "source_search_task_id": "old-mc",
        },
        {
            "task_id": "new-mc",
            "stage": "deep_search",
            "status": "completed",
            "model_version": "m1",
            "search_group_index": 2,
        },
    ]
    s["mlip_feedback_baselines"] = {"m1": deepcopy(s["phase_diagrams"]["mlip"])}
    record_feedback_cycle(s)
    assert not s["feedback_cycles"]


def test_advisory_recommendation_does_not_truncate_to_reference_count():
    from phase_agent.decisions.qbc_selection.recommend_post_mc_dft import recommend_post_mc_dft

    pool = [{"candidate_id": str(i), "atom_count": 2} for i in range(4)]
    c = default_layered_search_config()
    c["dft"]["selection"]["single_point_max_per_round"] = 1
    assert len(recommend_post_mc_dft(pool, c)["selected_candidates"]) == 4


def test_advisory_old_dft_count_cap_does_not_block_approved_sampling():
    from phase_agent.tools.policy.workflow_policy import action_errors

    c = default_layered_search_config()
    c["workflow"]["finetune_min_new_dft"] = 1
    c["workflow"]["finetune_max_new_dft"] = 1
    a = {
        "tool": "select_dft_candidates",
        "parameters": {
            "decisions": [
                {"candidate_id": "a", "action": "DFT_SINGLE_POINT"},
                {"candidate_id": "b", "action": "DFT_SINGLE_POINT"},
            ]
        },
    }
    assert not action_errors(a, {}, c)


def test_real_ui_supercell_size_max_is_deferred_until_tasks_end():
    from phase_agent.configuration.runtime.authorize_budget_extension import (
        authorize_budget_extension,
    )

    old = {"system": {"H_generation": {"enabled": True, "size_max": 16}}}
    new = deepcopy(old)
    new["system"]["H_generation"]["size_max"] = 20
    s = {
        "confirmed_config": old,
        "confirmed_config_version": "old",
        "tasks": [{"task_id": "t", "status": "running", "stage": "deep_search"}],
    }
    snapshot = {"config_version": "new", "config": new}
    result = authorize_budget_extension(s, snapshot)
    assert result["status"] == "deferred_supercell_revision"
    assert result["state"]["confirmed_config"] == old
    s["tasks"][0]["status"] = "completed"
    assert authorize_budget_extension(s, snapshot)["status"] == "supercell_revision_applied"


def test_agent_official_hull_and_gain_context_excludes_dft_and_old_models():
    from phase_agent.analysis.state.build_state_snapshot import build_state_snapshot

    s = {
        "phase_diagrams": {
            "mlip": {"energy_basis_id": "current", "entries": []},
            "dft": {"entries": [{"ehull": -123, "is_stable": True}]},
            "combined": {"entries": []},
        },
        "rewards": [
            {"energy_method": "dft", "reward": 999, "energy_basis_id": "current"},
            {"energy_method": "mlip", "reward": 888, "energy_basis_id": "old"},
            {"energy_method": "mlip", "reward": 1, "energy_basis_id": "current"},
        ],
    }
    view = build_state_snapshot(s, snapshot_index=1)
    assert set(view["current_convex_hull"]) == {"mlip"}
    assert set(view["decision_context"]["current_phase_diagram"]) == {"mlip"}
    assert [r["reward"] for r in view["short_term_memory"]["recent_rewards"]] == [1]
    assert [r["reward"] for r in view["decision_context"]["recent_experience"]["rewards"]] == [1]


def test_direct_training_preparation_also_waits_for_actual_tasks(tmp_path):
    from phase_agent.tools.workflows.prepare_remote_finetune import prepare_remote_finetune

    s = {
        "tasks": [
            {
                "task_id": "t",
                "stage": "dft_single_point",
                "status": "running",
                "recovery_wait_waived": True,
            }
        ]
    }
    result = prepare_remote_finetune(s, {"upload_batches_directory": str(tmp_path / "not-created")})
    assert result["status"] == "tasks_awaiting_results"
    assert not (tmp_path / "not-created").exists()


def test_production_hull_mode_does_not_build_or_export_a_dft_hull(tmp_path):
    from phase_agent.analysis.phase.update_phase_diagram import update_phase_diagram

    result = update_phase_diagram([], output_directory=tmp_path, include_dft=False)
    assert "dft" not in result["diagrams"]
    assert not list(tmp_path.rglob("*dft*"))


def test_confirmed_shared_choice_reuses_runtime_storage_identity(tmp_path):
    from phase_agent.persistence.shared_dft import settings, select

    s = {
        "shared_dft": {
            "directory": str(tmp_path / "store"),
            "diagram_id": "here",
            "enabled": False,
            "source_projects": [],
        }
    }
    c = {"shared_dft": {"enabled": True, "source_projects": ["named-project"]}}
    options = settings(s, c)
    assert options["directory"] == str(tmp_path / "store")
    assert options["diagram_id"] == "here"
    assert options["enabled"] is True
    assert select(s, c) == []
    assert not (tmp_path / "store").exists()

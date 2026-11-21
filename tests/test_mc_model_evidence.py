"""Model-specific MC evidence and historical energy-basis regression."""

from copy import deepcopy
import pytest
from phase_agent.analysis.phase.mc_snapshot import mc_model_snapshot
from phase_agent.science.mc.second_round_state import (
    first_round_source,
    has_relevant_mc_tasks,
    legacy_committee_task_keys,
)
from phase_agent.science.mc.schedule_tiered_mc import schedule_tiered_mc


def example():
    task = {
        "stage": "deep_search",
        "status": "completed",
        "task_id": "old",
        "task_key": "old-key",
        "branch_id": "B",
        "model_version": "m1",
        "outputs": {
            "energy_per_atom": -2.0,
            "energy": -8.0,
            "qbc": {"member_count": 4},
            "pool_summary": [{"energy_mean_per_atom": -2.0}],
        },
    }
    record = {
        "record_id": "r",
        "source_task_id": "old",
        "structure_id": "s",
        "structure_sha256": "a" * 64,
        "energy": -8.0,
    }
    frame = {
        "record_id": "r",
        "source_task_id": "old",
        "structure_id": "s",
        "structure_sha256": "a" * 64,
    }
    return {
        "active_model_version": "m1",
        "tasks": [task],
        "phase_records": [record],
        "mc_energy_repairs": {
            "fix": {
                "approval_record_id": "approved",
                "preview": {"checksum": "fix", "frames": [frame]},
            }
        },
        "tiered_mc_state": {
            "segments": [
                {
                    "task_key": "old-key",
                    "branch_id": "B",
                    "model_version": "m1",
                    "status": "completed",
                    "segment_index": 0,
                }
            ]
        },
    }


def test_model_hull_rejects_mixed_fallback_and_checks_independent_provenance():
    mixed = {
        "model_version": "m1",
        "mixed_model_versions": True,
        "source_model_versions": ["m0", "m1"],
    }
    pure = {"model_version": "m1", "version": "current", "entries": [{"source_version": "m1"}]}
    state = {"phase_diagrams": {"mlip": mixed}, "phase_diagrams_by_model": {"m1": pure}}
    assert mc_model_snapshot(state, "m1") is pure
    del state["phase_diagrams_by_model"]
    with pytest.raises(ValueError, match="independent_hull_required"):
        mc_model_snapshot(state, "m1")
    state["phase_diagrams_by_model"] = {"m1": {**pure, "entries": [{"source_version": "m0"}]}}
    with pytest.raises(ValueError, match="independent_hull_required"):
        mc_model_snapshot(state, "m1")


def test_old_committee_history_excluded_only_when_repair_bound_and_proven():
    state = example()
    assert legacy_committee_task_keys(state) == {"old-key"}
    assert first_round_source(state, "m1") is None
    assert not has_relevant_mc_tasks(state)
    state["tasks"].append(
        {"stage": "deep_search", "status": "completed", "model_version": "m0", "task_key": "other"}
    )
    assert not has_relevant_mc_tasks(state)
    state["tasks"][-1]["status"] = "running"
    assert has_relevant_mc_tasks(state)
    tampered = example()
    tampered["mc_energy_repairs"]["fix"]["preview"]["frames"][0]["structure_sha256"] = "b" * 64
    assert not legacy_committee_task_keys(tampered)
    assert first_round_source(tampered, "m1")["completed_count"] == 1
    main = example()
    main["tasks"][0]["outputs"]["energy_source"] = "main_model_selected_frame"
    assert not legacy_committee_task_keys(main)


def test_schedule_keeps_history_but_new_model_search_starts_at_zero():
    state = example()
    original = deepcopy(state["tiered_mc_state"])
    result = schedule_tiered_mc(
        [{"branch_id": "B", "P": "O3", "x": "1/2", "atom_count": 4, "relaxed_ehull": 0.0}],
        state["tiered_mc_state"],
        policy={
            "tiers": [
                {"name": "base", "max_mc_steps": 10, "patience_steps": 3, "min_improvement": 0.001}
            ],
            "minimum_continuation_tasks": 0,
        },
        total_budget=100,
        seed=1,
        model_version="m1",
        hull_reference_version="h",
        scope_by_model=True,
        excluded_history_task_keys=legacy_committee_task_keys(state),
    )
    assert result["actions"][0]["segment_index"] == 0
    assert result["state"]["segments"][0] == original["segments"][0]
    assert len(result["state"]["segments"]) == 2
    state["tiered_mc_state"]["segments"][0]["status"] = "running"
    result = schedule_tiered_mc(
        [],
        state["tiered_mc_state"],
        policy={
            "tiers": [
                {"name": "base", "max_mc_steps": 10, "patience_steps": 3, "min_improvement": 0.001}
            ]
        },
        total_budget=100,
        seed=1,
        model_version="m1",
        hull_reference_version="h",
        scope_by_model=True,
        excluded_history_task_keys={"old-key"},
    )
    assert result["status"] == "tasks_in_progress"


def test_standalone_mc_limit_is_per_batch_not_lifetime():
    from phase_agent.configuration.defaults.default_layered_search_config import (
        default_layered_search_config,
    )
    from phase_agent.policies.validate_tool_action import validate_tool_action

    config = default_layered_search_config()
    config["budgets"]["stage_limits"]["deep_search"]["max_tasks"] = 300
    session = {
        "status": "confirmed",
        "confirmed_snapshot": {"config": config, "config_version": "c"},
    }
    state = {"budget_usage": {"stages": {"deep_search": {"tasks": 610}}}}
    registry = {"allocate_mc_bohb": {"handler": lambda **kwargs: None}}
    action = {
        "tool": "allocate_mc_bohb",
        "stage": "deep_search",
        "budget": 1,
        "parameters": {"mc_budget": 10},
        "target_ids": [f"B{i}" for i in range(500)],
    }
    result = validate_tool_action(action, state, session, registry)
    assert "stage_task_limit" not in result["errors"]
    action["target_ids"].append("B500")
    result = validate_tool_action(action, state, session, registry)
    assert "relax_mc_branch_limit_exceeded" in result["errors"]
    action["target_ids"] = ["B1", "B1"]
    result = validate_tool_action(action, state, session, registry)
    assert "relax_mc_branch_scope_invalid" in result["errors"]
    action["target_ids"] = []
    result = validate_tool_action(action, state, session, registry)
    assert "stage_task_limit" in result["errors"]


def test_frozen_pool_uses_screened_structure_without_generating_relax(tmp_path):
    from types import SimpleNamespace
    from phase_agent.tools.workflows.prepare_branch_relaxation import prepare_branch_relaxation
    from phase_agent.tools.remote.integrity import file_checksum

    path = tmp_path / "frame.vasp"
    path.write_text("frozen frame")
    record = {
        "branch_id": "B",
        "structure_id": "S",
        "structure_path": str(path),
        "structure_sha256": file_checksum(path),
        "composition": {"Na": 1, "Co": 1, "O": 2},
        "energy": -4.0,
        "energy_unit": "eV",
        "model_version": "m1",
        "stage": "relax_and_feature",
        "converged": True,
    }
    pool = {"version": "pool", "model_version": "m1", "records": [record]}
    entry = {
        "structure_id": "S",
        "structure_path": str(path),
        "normalized_total_energy": -4.0,
        "phase_identification_status": "identified",
        "phase": "O3",
        "ehull": 0.01,
        "ehull_unit": "eV/atom",
    }
    diagram = {
        "status": "completed",
        "method": "mlip",
        "model_version": "m1",
        "version": "h",
        "entries": [entry],
    }
    state = {
        "tasks": [],
        "branch_hull_batches": {"pool": pool},
        "phase_diagrams_by_model": {"m1": diagram},
    }
    context = {
        "effective_config": {"bohb": {}, "mlip": {"version": "m1"}},
        "manager": SimpleNamespace(data={}),
        "mc_hull_reference_version": "pool",
        "mc_phase_diagram_version": "h",
    }
    candidates = [{"branch_id": "B", "P": "O3", "x": "1/2"}]
    result = prepare_branch_relaxation(candidates, state, context)
    assert result["status"] == "ready" and result["candidates"][0]["structure_id"] == "S"
    assert result["state"]["tasks"] == [] and state["tasks"] == []
    path.write_text("modified")
    result = prepare_branch_relaxation(candidates, state, context)
    assert result["reason"] == "approved_structure_hash_changed" and not result["state"]["tasks"]
    path.unlink()
    result = prepare_branch_relaxation(candidates, state, context)
    assert (
        result["reason"] == "approved_structure_or_phase_unavailable"
        and not result["state"]["tasks"]
    )


def test_cancel_unsent_screening_is_bound_and_preserves_other_tasks():
    from phase_agent.tools.state.cancel_unsent_mc_screening import cancel_unsent_mc_screening

    identity = {"invocation_id": "d"}
    task = {
        "task_id": "t",
        "task_key": "k",
        "stage": "relax_and_feature",
        "status": "pending",
        "model_version": "m",
        "config_version": "c",
        "parent_decision_id": "d",
        "branch_id": "b",
    }
    refs = [{k: v for k, v in task.items() if k != "branch_id"}]
    action = {
        "tool": "allocate_mc_bohb",
        "target_ids": ["b"],
        "parameters": {"budget_preview": {"allocation_checksum": "hash"}},
    }
    events = [
        {"event": "claimed", "payload": {"identity": identity, "action": action}},
        {
            "event": "tool_returned",
            "payload": {
                "identity": identity,
                "outcome": "completed",
                "evidence": {"result_status": "completed", "returned_task_refs": refs},
            },
        },
    ]
    state = {
        "tasks": [
            task,
            {"task_id": "other", "task_key": "other", "stage": "deep_search", "status": "pending"},
        ],
        "branch_batch": {"status": "relax_pending", "branch_ids": ["b"], "history": []},
        "pending_branch_screening": {"branch_ids": ["b"]},
        "budget_reservations": {
            "k": {"status": "reserved", "stage": "relax_and_feature", "reserved_cost": 2}
        },
    }
    before = deepcopy(state)
    result, count = cancel_unsent_mc_screening(
        state, events, decision_id="d", allocation_checksum="hash"
    )
    assert count == 1 and result["tasks"][0]["status"] == "cancelled"
    assert result["budget_reservations"]["k"]["status"] == "released"
    assert [x["task_id"] for x in result["pending_tasks"]] == ["other"]
    assert result["branch_batch"]["status"] == "failed" and state == before
    for key, value in (
        ("slurm_batch_id", "123"),
        ("input_path", "somefile"),
        ("status", "running"),
    ):
        altered = deepcopy(state)
        altered["tasks"][0][key] = value
        with pytest.raises(ValueError):
            cancel_unsent_mc_screening(altered, events, decision_id="d", allocation_checksum="hash")
    with pytest.raises(ValueError):
        cancel_unsent_mc_screening(state, events, decision_id="d", allocation_checksum="other")


def test_approved_frozen_pool_allocates_mc_without_new_screening(tmp_path):
    from tests.test_relax_mc_handoff import _completed_relax_fixture
    from phase_agent.tools.workflows.prepare_mc_preview import prepare_mc_preview
    from phase_agent.tools.workflows.create_active_learning_handlers import _allocate_mc_bohb

    manager, state, config = _completed_relax_fixture(tmp_path)
    state["tasks"] = []
    state["budget_usage"] = {"stages": {"deep_search": {"tasks": 610}}}
    config["budgets"]["stage_limits"]["deep_search"]["max_tasks"] = 300
    action = {
        "tool": "allocate_mc_bohb",
        "task_key": "mc-test",
        "target_ids": ["B1"],
        "parameters": {
            "mc_budget": 900,
            "branch_count": 1,
            "seed": 42,
            "exploration_fraction": 0.1,
        },
        "budget": 0.0,
    }
    action = prepare_mc_preview(action, state, {"manager": manager}, config)
    result = _allocate_mc_bohb(
        action=action,
        context={
            "manager": manager,
            "phase_references": {},
            "effective_config": config,
            "event_state": state,
            "config_version": "v1",
            "execution_mode": "testing",
        },
    )
    assert result["status"] == "completed"
    assert len(result["actions"]) == 1 and result["actions"][0]["stage"] == "deep_search"
    assert not any(row.get("stage") == "relax_and_feature" for row in result["state"]["tasks"])

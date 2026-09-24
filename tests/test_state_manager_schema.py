from config_layer.schema.action_state_schema import ACTION_REQUIRED_FIELDS, STATE_REQUIRED_FIELDS
from data_layer.memory.decision_memory import update_long_term_memory
from execution_layer.dispatch.create_tool_registry import create_tool_registry
from execution_layer.workflows.run_tool_step import run_tool_step
from execution_layer.state.state_manager import update_state_snapshot
from tests.test_event_loop import _session
from config_layer.schema.state_schema_migrations import migrate_state_snapshot
import pytest


def test_snapshot_contains_required_scientific_and_runtime_summaries():
    state = {
        "status": "searching", "budget_remaining": 12,
        "coverage": {"fraction": .5},
        "phase_diagrams": {"mlip": {"version": "h1", "entries": [
            {"record_id": "S1", "composition": {"Na": 1}, "ehull": 0, "is_stable": True}
        ]}},
        "qbc_candidates": [{"candidate_id": "S2", "qbc": {"f_std_max": .3}}],
        "active_model_version": "m1",
        "action_records": [{"record_id": "a1", "status": "completed",
                            "final_action": {"tool": "check_convergence", "reason": "check"}}],
    }
    updated = update_state_snapshot(state, config_version="v1")
    snapshot = updated["current_state_snapshot"]
    assert all(field in snapshot for field in STATE_REQUIRED_FIELDS)
    assert snapshot["current_convex_hull"]["mlip"]["stable_entries"][0]["record_id"] == "S1"
    assert snapshot["search_coverage"]["fraction"] == .5
    assert snapshot["uncertainty"][0]["f_std_max"] == .3
    assert snapshot["mlip_status"]["active_version"] == "m1"


def test_agent_receives_snapshot_only_and_action_is_canonical():
    payloads = []
    registry = create_tool_registry({"check_convergence": lambda **_: {"status": "completed"}})
    result = run_tool_step(
        {"tasks": [{"task_id": "private-task", "status": "completed"}]}, _session(),
        registry=registry, execute=False,
        agent_client=lambda payload: payloads.append(payload) or {
            "tool": "check_convergence", "parameters": {}, "budget": 0, "reason": "check"
        },
    )
    visible = payloads[0]["state"]
    assert all(field in visible for field in STATE_REQUIRED_FIELDS)
    assert "tasks" not in visible
    assert all(field in result["action"] for field in ACTION_REQUIRED_FIELDS)


def test_short_term_failures_do_not_modify_long_term_memory():
    state = update_long_term_memory({}, {
        "physical_priors": ["固定 TM 组成"],
        "search_rules": ["优先近凸包区域"],
    }, source="user")
    state["tasks"] = [{"task_id": "bad", "status": "failed", "error": "scheduler"}]
    updated = update_state_snapshot(state)
    memory = updated["decision_memory"]
    assert memory["long_term"]["physical_priors"] == ["固定 TM 组成"]
    assert memory["short_term"]["failed_tasks"][0]["task_id"] == "bad"
    assert "scheduler" not in memory["long_term"]["physical_priors"]


def test_v0_snapshot_migrates_and_future_version_is_rejected():
    migrated = migrate_state_snapshot({"status": "running", "remaining_budget": 5,
                                       "qbc_uncertainty": [], "recent_action_history": []})
    assert migrated["schema_version"] == 1
    assert migrated["current_status"] == "running"
    assert migrated["available_budget"] == 5
    with pytest.raises(ValueError, match="future state schema"):
        migrate_state_snapshot({"schema_version": 99})

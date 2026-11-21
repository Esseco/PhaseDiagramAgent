from copy import deepcopy

from phase_agent.runtime.pending_plan_state import (
    stale_pending_plan_ids,
    supersede_stale_pending,
    visible_pending,
)


def test_context_changed_plan_is_hidden_and_superseded(monkeypatch):
    monkeypatch.setattr(
        "phase_agent.evidence.project_authority.project_identity",
        lambda state: {"model_version": "new", "config_version": "new"},
    )
    state = {
        "pending_execution_policies": {
            "stale": {"project_identity": {"model_version": "old", "config_version": "old"}},
            "current": {"project_identity": {"model_version": "new", "config_version": "new"}},
        },
        "action_records": [
            {"record_id": "stale", "status": "awaiting_approval"},
            {"record_id": "current", "status": "awaiting_approval"},
        ],
    }
    assert stale_pending_plan_ids(state) == ["stale"]
    assert list(visible_pending(state)) == ["current"]
    before = deepcopy(state)
    removed, changed = supersede_stale_pending(state)
    assert changed and removed[0]["plan_id"] == "stale"
    assert "stale" not in state["pending_execution_policies"]
    assert removed[0]["plan"] == before["pending_execution_policies"]["stale"]
    assert state["action_records"][0]["status"] == "superseded"
    assert state["pending_execution_policies"]["current"] == before["pending_execution_policies"]["current"]

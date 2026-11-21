from copy import deepcopy

import pytest

from phase_agent.runtime.codex.team import call_team_tool
from tests.test_codex_native_team import advice, make_control, team


def test_mcp_status_read_does_not_rebuild_evidence(tmp_path, monkeypatch):
    control, _ = make_control(tmp_path)
    t = team(control)
    original = t.start(question="Review the saved action", consult=["exploration"])

    def unexpected(*args, **kwargs):
        raise AssertionError("status read rebuilt scientific evidence")

    monkeypatch.setattr(type(t), "_context", unexpected)
    response = call_team_tool(control, "phase_team_read", {})
    assert response["team_id"] == original["team_id"]
    assert response["remaining_roles"] == ["exploration"]
    assert response["stale"] is None and response["execution_approved"] is None
    assert "snapshot" not in response
    with pytest.raises(AssertionError, match="rebuilt"):
        call_team_tool(control, "phase_team_read", {"include_details": True})
    with pytest.raises(ValueError, match="human_review_requires_coordinator_draft"):
        t.feedback(original["team_id"], 1, "accept", "approve")
    with pytest.raises(ValueError, match="simulation_requires_current_strategic_draft"):
        t.simulate(original["team_id"], 1, {})


def test_revise_binds_new_role_tasks_before_experts_answer(tmp_path, monkeypatch):
    control, context = make_control(tmp_path)
    t = team(control)

    def with_role_scope(*, question=None, role_tasks=None, **kwargs):
        from phase_agent.persistence.memory.decision_context import (
            current_memory_query,
            memory_query_scope,
        )

        with memory_query_scope(question, role_tasks):
            return {**deepcopy(context), "context_digest": current_memory_query()}

    monkeypatch.setattr(t, "_context", with_role_scope)
    original = t.start(question="Review sampling", consult=["exploration"])
    new = t.feedback(
        original["team_id"],
        1,
        "revise",
        "Use the new scope",
        role_tasks={
            "exploration": {
                "question": "Which composition needs more coverage?",
                "completion_criteria": ["Cite coverage evidence"],
            },
        },
    )
    assert new["stale"] is False
    current = t.read(original["team_id"], request_role="exploration")
    assert current["stale"] is False
    assert current["snapshot"]["snapshot_id"] == new["snapshot"]["snapshot_id"]
    assert t.advise(original["team_id"], 2, advice(new), compact=True)["remaining_roles"] == []


def test_mcp_direction_accept_returns_acknowledgement_only(tmp_path):
    from tests.test_codex_native_team import prepared

    control, _, t, view = prepared(tmp_path)
    params = {"team_id": view["team_id"], "revision": 1}
    t.draft(**params, action={"tool": "pause_search"}, rationale="review")
    response = call_team_tool(
        control,
        "phase_team_feedback",
        {
            **params,
            "decision": "accept",
            "feedback": "I accept the direction",
        },
    )
    assert response["status"] == "ready_to_propose"
    assert response["stale"] is False and response["execution_approved"] is False
    assert "snapshot" not in response

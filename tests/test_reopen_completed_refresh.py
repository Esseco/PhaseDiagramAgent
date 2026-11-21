from copy import deepcopy

import pytest

from phase_agent.tools.workflows.reopen_completed_refresh import reopen_state


@pytest.fixture
def example():
    baseline = {"status": "completed", "version": "base", "model_version": "old"}
    refresh = {"status": "completed", "new_model_version": "new", "task_ids": ["t"], "plan": {"checksum": "old-plan"}}
    cycle = {"status": "completed", "replacement_model_version": "new", "replacement_hull_version": "hull", "baseline_model_version": "old", "baseline_hull_version": "base", "baseline_diagram": baseline}
    state = {"active_model_version": "new", "model_refresh": refresh, "extrapolation_recovery_cycle": cycle,
             "phase_diagrams": {"mlip": {"version": "hull", "entries": [{"record_id": "r"}]}},
             "tasks": [{"task_id": "t", "status": "completed"}]}
    preview = {"status": "preview_only", "submitted": False, "prior_plan_checksum": "old-plan", "prior_hull_version": "hull", "plan": {"checksum": "new-plan", "new_model_version": "new", "baseline_hull_version": "base", "hd_refresh_reference": {"hull_version": "base"}}}
    return state, preview


def call(state, preview):
    return reopen_state(state, preview, expected_model="new", expected_plan="old-plan", expected_hull="hull", expected_preview="new-plan")


def test_reopen_preserves_prior_data_and_starts_unapproved_attempt(example):
    state, preview = example
    before = deepcopy(state)
    result = call(state, preview)
    assert state == before
    assert result["model_refresh"]["status"] == "approval_required"
    assert "task_ids" not in result["model_refresh"]
    assert result["tasks"] == before["tasks"]
    assert result["phase_diagrams"]["mlip"]["status"] == "not_ready"
    assert result["completed_refresh_replans"][0]["prior_refresh"] == before["model_refresh"]
    assert result["completed_refresh_replans"][0]["prior_hull"] == before["phase_diagrams"]["mlip"]
    assert result["extrapolation_recovery_cycle"]["status"] == "awaiting_new_model_refresh"


@pytest.mark.parametrize("edit", [
    lambda s,p: s.update(active_model_version="wrong"),
    lambda s,p: s.update(pending_execution_policies={"x": {}}),
    lambda s,p: s["model_refresh"].update(status="approval_required"),
    lambda s,p: s["phase_diagrams"]["mlip"].update(version="wrong"),
    lambda s,p: p["plan"].update(checksum="wrong"),
    lambda s,p: p["plan"].update(hd_refresh_reference=None),
    lambda s,p: s["extrapolation_recovery_cycle"].update(baseline_hull_version="wrong"),
])
def test_reopen_fails_closed_on_changed_binding(example, edit):
    state, preview = example
    edit(state, preview)
    with pytest.raises(ValueError):
        call(state, preview)


def test_new_round_does_not_count_previous_refresh_tasks(example):
    from phase_agent.agents.reliability.extrapolation_round import assess_current_rounds
    state, preview = example
    reopened = call(state, preview)
    assert assess_current_rounds(reopened)["rounds"] == []


def test_mcp_dispatches_bound_reopen_request():
    from phase_agent.runtime.codex.mcp import CodexMcpServer
    class FakeCodex:
        def reopen_completed_refresh(self, **kwargs):
            return {"status": "reopened", "bindings": kwargs}
    args = {"expected_model": "new", "expected_plan": "old-plan", "expected_hull": "hull", "expected_preview": "new-plan"}
    assert CodexMcpServer(FakeCodex())._call_tool("phase_reopen_completed_refresh", args)["bindings"] == args


def test_existing_pending_proposal_finishes_checkpoint_before_return(monkeypatch):
    from threading import RLock
    from types import SimpleNamespace
    from unittest.mock import Mock
    from phase_agent.runtime.local_agent_control import LocalAgentControl

    checksum = "digest"
    invocation = "reopened-refresh-" + checksum
    handler = SimpleNamespace(
        state_path="ignored", lock=RLock(), workflow_kwargs={},
        _run=Mock(return_value={"status": "awaiting_approval"}),
    )
    state = {
        "model_refresh": {"status": "approval_required", "reopened_preview_checksum": checksum},
        "pending_execution_policies": {invocation: {}},
    }
    monkeypatch.setattr("phase_agent.runtime.local_agent_control.read_json", lambda *args: state)
    result = LocalAgentControl(handler).propose_reopened_refresh(expected_preview=checksum)
    assert result["status"] == "awaiting_approval"
    handler._run.assert_called_once_with(invocation, None, "按已冻结范围提出本轮刷新审批")


def test_reopened_refresh_rejects_changed_preview_before_preparation(tmp_path):
    from tests.test_model_refresh_workflow import setup
    from phase_agent.tools.workflows.model_refresh_gate import run_model_refresh_gate
    state, session, context, registry = setup(tmp_path)
    state["model_refresh"]["reopened_preview_checksum"] = "different-preview"
    result = run_model_refresh_gate(state, session, registry=registry, context=context)
    assert result["status"] == "not_configured"
    assert result["submitted"] is False
    assert "重开刷新方案已变化" in result["reason"]
    assert not (tmp_path / "submissions").exists()


def test_refresh_reason_preserves_matching_pending_action(tmp_path):
    from tests.test_model_refresh_workflow import setup
    from phase_agent.tools.workflows.model_refresh_gate import run_model_refresh_gate
    state, session, context, registry = setup(tmp_path)
    first = run_model_refresh_gate(state, session, registry=registry, context=context)
    assert first["status"] == "awaiting_approval"
    pending = first["state"]["pending_execution_policies"]["__single_interactive_action__"]
    pending["agent_proposal"]["raw_action"]["reason"] = "frozen wording"
    again = run_model_refresh_gate(first["state"], session, registry=registry, context=context)
    assert again["status"] == "awaiting_approval"
    assert again["action"]["reason"] == "frozen wording"

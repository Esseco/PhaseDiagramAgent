from types import SimpleNamespace
from phase_agent.graphs.dialogue.support import _save_binding, _load_binding


def test_binding_survives_handler_recreation_and_isolates_threads(tmp_path):
    a = SimpleNamespace(state_path=tmp_path / "state.json", conversation_id="a")
    binding = {"plan_id": "p", "state_version": "v", "proposal_hash": "h"}
    _save_binding(a, binding)
    restored = SimpleNamespace(state_path=a.state_path, conversation_id="a")
    assert _load_binding(restored) == binding
    other = SimpleNamespace(state_path=a.state_path, conversation_id="b")
    assert _load_binding(other) is None
    _save_binding(restored, None)
    assert _load_binding(a) is None


def test_second_approval_calls_review_after_handler_recreation(tmp_path, monkeypatch):
    from phase_agent.graphs.dialogue.support import review_presented_proposal
    from unittest.mock import Mock
    state = {"pending_execution_policies": {"p": {"agent_proposal": {"recommended_action": "generate_branches"}}}}
    monkeypatch.setattr("phase_agent.graphs.dialogue.support.build_status_summary", lambda *a, **k: {"summary_id": "v"})
    monkeypatch.setattr("phase_agent.graphs.dialogue.support.proposal_hash", lambda p: "h")
    monkeypatch.setattr("phase_agent.runtime.chat_approval_rules.is_sensitive_proposal", lambda p: False)
    monkeypatch.setattr("phase_agent.runtime.plan_queries.pending_plan_reply", lambda state: "saved plan")
    monkeypatch.setattr("phase_agent.runtime.workflow_reply_presentation.format_workflow_reply", lambda *a: "executed once")
    first = SimpleNamespace(state_path=tmp_path / "state.json", conversation_id="a", review_pending=Mock())
    assert "saved plan" in review_presented_proposal(first, state, "同意")
    first.review_pending.assert_not_called()
    second = SimpleNamespace(state_path=first.state_path, conversation_id="a", review_pending=Mock(return_value={"result": {}}))
    assert review_presented_proposal(second, state, "同意") == "executed once"
    second.review_pending.assert_called_once_with("p", "approve", expected_state_version="v", expected_proposal_hash="h", comment="同意")
    assert _load_binding(first) is None

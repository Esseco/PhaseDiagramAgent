from unittest.mock import Mock, patch

from phase_agent.runtime.codex.offline import CodexOffline


def test_offline_reuses_existing_gates_and_binds_context():
    handler = Mock()
    control = Mock()
    control.codex_context.return_value = {"context_version": "v1", "context_digest": "d1"}
    control.propose_codex.return_value = {"status": "awaiting_approval"}
    with patch("phase_agent.runtime.codex.offline.LocalAgentControl", return_value=control):
        agent = CodexOffline("runtime.json", runtime_factory=Mock(return_value=handler))
        action = {"tool": "generate_branches"}
        assert agent.propose(action)["status"] == "awaiting_approval"
        control.propose_codex.assert_called_once_with(
            action, context_version="v1", context_digest="d1", conversation_id="codex-offline"
        )
        control.decide.assert_not_called()


def test_confirmation_requires_explicit_and_reloads_only_after_success():
    factory = Mock()
    control = Mock()
    control.confirm_config.side_effect = [{"status": "draft"}, {"status": "confirmed"}]
    with patch("phase_agent.runtime.codex.offline.LocalAgentControl", return_value=control):
        agent = CodexOffline("runtime.json", runtime_factory=factory)
        agent.confirm_config()
        assert factory.call_count == 1
        control.confirm_config.assert_called_with(explicit=False)
        agent.confirm_config(explicit=True)
        assert factory.call_count == 2


def test_decision_preserves_frozen_plan_bindings():
    control = Mock()
    with patch("phase_agent.runtime.codex.offline.LocalAgentControl", return_value=control):
        agent = CodexOffline("runtime.json", runtime_factory=Mock())
        agent.decide("approve", plan_id="p", state_version="v", proposal_hash="h")
        control.decide.assert_called_once_with(
            "approve",
            plan_id="p",
            expected_state_version="v",
            expected_proposal_hash="h",
            comment="",
            conversation_id="codex-offline",
        )


def test_saved_proposal_is_not_retried_when_pending_lookup_fails():
    control = Mock()
    control.codex_context.return_value = {"context_version": "v1", "context_digest": "d1"}
    control.propose_codex.return_value = {"reply": {"status": "awaiting_approval"}}
    control.pending.side_effect = OSError("temporarily unavailable")
    with patch("phase_agent.runtime.codex.offline.LocalAgentControl", return_value=control):
        agent = CodexOffline("runtime.json", runtime_factory=Mock())
        result = agent.propose({"tool": "generate_branches"})
        assert result["reply"]["status"] == "awaiting_approval"
        assert result["pending_error"] == "OSError"
        control.propose_codex.assert_called_once()

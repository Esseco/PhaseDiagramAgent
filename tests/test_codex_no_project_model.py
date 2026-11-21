from types import SimpleNamespace
from unittest.mock import Mock, patch
import threading
import pytest

from phase_agent.runtime.chat_review import review_pending
from phase_agent.runtime.codex.offline import CodexOffline
from phase_agent.runtime.codex.control import CodexControlClient
from phase_agent.policies.file_approval import proposal_hash


def fixture_handler(source="codex"):
    original = Mock(side_effect=AssertionError("DeepSeek was called"))
    proposal = {"raw_action": {"tool": "pause_search", "decision_source": source}}
    state = {"pending_execution_policies": {"p": {"agent_proposal": proposal}}}
    handler = SimpleNamespace(
        lock=threading.RLock(), state_path="ignored", workflow_kwargs={"agent_client": original}
    )
    return handler, original, state, proposal_hash(proposal)


def call_review(handler, state, digest, decision="approve", operator="agent"):
    with (
        patch("phase_agent.runtime.chat_review.read_json", return_value=state) as read_state,
        patch(
            "phase_agent.runtime.chat_review.build_status_summary", return_value={"summary_id": "v"}
        ),
        patch(
            "phase_agent.evidence.project_authority.project_identity",
            return_value={"model_consistent": True},
        ),
        patch("phase_agent.policies.recovery_transition.post_recovery_tools", return_value=None),
        patch(
            "phase_agent.policies.recovery_transition.legacy_post_recovery_plan_ids",
            return_value=set(),
        ),
    ):
        result = review_pending(
            handler,
            "p",
            decision,
            expected_state_version="v",
            expected_proposal_hash=digest,
            comment="change",
            is_sensitive=lambda p: False,
            operator=operator,
        )
        read_state.assert_called_once_with(handler.state_path, {})
        return result


@pytest.mark.parametrize("source,operator", [("codex", "agent"), ("agent", "codex")])
def test_approval_blocks_project_model_and_restores_original(source, operator):
    handler, original, state, digest = fixture_handler(source)

    def run(*args):
        client = handler.workflow_kwargs["agent_client"]
        assert client is not original
        from phase_agent.agents.simulation.agent import inspect_execution

        report = inspect_execution(
            {"tool": "pause_search"}, {"status": "failed", "result": {"status": "failed"}}, client
        )
        assert "diagnosis_request" in report and "diagnosis" not in report
        return {"status": "completed"}

    handler._run = run
    assert call_review(handler, state, digest, operator=operator)["status"] == "completed"
    assert handler.workflow_kwargs["agent_client"] is original
    original.assert_not_called()


def test_codex_modify_requires_codex_revision_without_running_workflow():
    handler, original, state, digest = fixture_handler()
    handler._run = Mock()
    with pytest.raises(ValueError, match="Codex"):
        call_review(handler, state, digest, decision="modify")
    handler._run.assert_not_called()
    original.assert_not_called()


def test_normal_agent_approval_keeps_original_client():
    handler, original, state, digest = fixture_handler("agent")
    handler._run = lambda *args: (
        {"status": "completed"}
        if handler.workflow_kwargs["agent_client"] is original
        else (_ for _ in ()).throw(AssertionError())
    )
    assert call_review(handler, state, digest)["status"] == "completed"


def test_codex_approval_restores_client_on_exception():
    handler, original, state, digest = fixture_handler()
    handler._run = Mock(side_effect=RuntimeError("simulated failure"))
    with pytest.raises(RuntimeError, match="simulated failure"):
        call_review(handler, state, digest)
    assert handler.workflow_kwargs["agent_client"] is original
    original.assert_not_called()


def test_offline_reload_installs_blocker_each_time():
    handlers = [
        SimpleNamespace(workflow_kwargs={"agent_client": Mock()}),
        SimpleNamespace(workflow_kwargs={"agent_client": Mock()}),
    ]
    originals = [h.workflow_kwargs["agent_client"] for h in handlers]
    with patch("phase_agent.runtime.codex.offline.LocalAgentControl"):
        agent = CodexOffline("unused", runtime_factory=Mock(side_effect=handlers))
        for i in range(2):
            if i:
                agent._load()
            client = agent.handler.workflow_kwargs["agent_client"]
            assert client is not originals[i]
            with pytest.raises(RuntimeError, match="Codex"):
                client({})
            originals[i].assert_not_called()


def test_http_codex_decision_carries_operator():
    client = CodexControlClient(operator="codex", control_token="local")
    with patch.object(CodexControlClient, "_control", return_value={}) as request:
        client.decide("approve", plan_id="p", state_version="v", proposal_hash="h")
    assert request.call_args.kwargs["payload"]["operator"] == "codex"

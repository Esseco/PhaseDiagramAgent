import io
import json
from unittest.mock import patch

from phase_agent.runtime.codex.control import CodexControlClient


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_status_uses_control_bearer_and_returns_json():
    calls = []

    def opener(request, timeout):
        calls.append((request, timeout))
        return FakeResponse({"status": "ok"})

    client = CodexControlClient("http://127.0.0.1:8765", control_token="control-secret")
    with patch("phase_agent.runtime.codex.control.urlopen", opener):
        result = client.status()

    assert result == {"status": "ok"}
    request, timeout = calls[0]
    assert request.full_url == "http://127.0.0.1:8765/phase/status"
    assert request.get_header("Authorization") == "Bearer control-secret"
    assert request.method == "GET"
    assert timeout == 30.0


def test_propose_posts_instruction_to_existing_control_endpoint():
    calls = []

    def opener(request, timeout):
        calls.append((request, timeout))
        return FakeResponse({"status": "pending_review"})

    client = CodexControlClient("http://127.0.0.1:8765", control_token="control-secret")
    with patch("phase_agent.runtime.codex.control.urlopen", opener):
        result = client.propose("继续项目", conversation_id="codex-1")

    assert result["status"] == "pending_review"
    request, _ = calls[0]
    assert request.full_url == "http://127.0.0.1:8765/phase/propose"
    assert request.method == "POST"
    assert json.loads(request.data) == {
        "instruction": "继续项目",
        "conversation_id": "codex-1",
    }


def test_decide_preserves_existing_optimistic_concurrency_fields():
    calls = []

    def opener(request, timeout):
        calls.append(request)
        return FakeResponse({"status": "completed"})

    client = CodexControlClient("http://127.0.0.1:8765", control_token="control-secret")
    with patch("phase_agent.runtime.codex.control.urlopen", opener):
        client.decide(
            "approve",
            plan_id="P1",
            state_version="S1",
            proposal_hash="H1",
            comment="继续",
        )

    assert json.loads(calls[0].data) == {
        "decision": "approve",
        "plan_id": "P1",
        "state_version": "S1",
        "proposal_hash": "H1",
        "comment": "继续",
        "conversation_id": "codex-control",
    }


def test_codex_operator_blocks_model_generating_requests():
    from phase_agent.runtime.codex.control import CodexControlError

    client = CodexControlClient(operator="codex", control_token="control-secret")
    try:
        client.continue_project()
    except CodexControlError as error:
        assert "不会调用 Agent/DeepSeek" in str(error)
    else:
        raise AssertionError("codex takeover must not call the Agent model")


def test_codex_operator_still_allows_existing_approval_decisions():
    calls = []

    def opener(request, timeout):
        calls.append(request)
        return FakeResponse({"status": "completed"})

    client = CodexControlClient(operator="codex", control_token="control-secret")
    with patch("phase_agent.runtime.codex.control.urlopen", opener):
        result = client.decide(
            "approve", plan_id="P1", state_version="S1", proposal_hash="H1"
        )

    assert result["status"] == "completed"
    assert calls[0].full_url.endswith("/phase/decision")


def test_codex_propose_posts_structured_action_to_dedicated_endpoint():
    calls = []

    def opener(request, timeout):
        calls.append(request)
        if request.full_url.endswith("/phase/codex/context"):
            return FakeResponse({"context_version": "V1", "context_digest": "D1"})
        return FakeResponse({"status": "awaiting_approval"})

    client = CodexControlClient(operator="codex", control_token="control-secret")
    action = {"tool": "pause_search", "reason": "Codex review"}
    with patch("phase_agent.runtime.codex.control.urlopen", opener):
        result = client.codex_propose(action, conversation_id="codex-cr")

    assert result["status"] == "awaiting_approval"
    assert calls[0].full_url.endswith("/phase/codex/context")
    assert calls[1].full_url.endswith("/phase/codex/propose")
    assert json.loads(calls[1].data) == {
        "action": action,
        "context_version": "V1",
        "context_digest": "D1",
        "conversation_id": "codex-cr",
    }


def test_codex_report_posts_only_existing_plan_id():
    calls = []

    def opener(request, timeout):
        calls.append(request)
        return FakeResponse({"report_path": "report.md"})

    client = CodexControlClient(operator="codex", control_token="control-secret")
    with patch("phase_agent.runtime.codex.control.urlopen", opener):
        assert client.codex_report("P1")["report_path"] == "report.md"
    assert calls[0].full_url.endswith("/phase/codex/report")
    assert json.loads(calls[0].data) == {"plan_id": "P1"}

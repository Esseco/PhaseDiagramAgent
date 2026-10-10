import pytest
from phase_agent.runtime.chat_application import RunWorkflowChatHandler, OpenWebUIRequestError
from phase_agent.runtime.studio_session_scope import studio_project_session


def test_studio_switches_thread_without_reusing_dialogue(tmp_path):
    handler = RunWorkflowChatHandler({"state_path": str(tmp_path / "state.json")})
    handler.conversation_id = "first"
    handler.recent_dialogue = [{"role": "user", "content": "first private context"}]
    with studio_project_session():
        reply = handler([{"role": "user", "content": "现在什么阶段"}], conversation_id="second")
    assert "当前" in reply
    assert all(row.get("content") != "first private context" for row in handler.recent_dialogue)


def test_legacy_http_single_session_guard_remains(tmp_path):
    handler = RunWorkflowChatHandler({"state_path": str(tmp_path / "state.json")})
    handler.conversation_id = "first"
    with pytest.raises(OpenWebUIRequestError, match="另一个"):
        handler([{"role": "user", "content": "现在什么阶段"}], conversation_id="second")


def test_each_studio_thread_restores_its_own_context(tmp_path, monkeypatch):
    handler = RunWorkflowChatHandler({"state_path": str(tmp_path / "state.json")})
    received = []
    def respond(messages, *, conversation_id=None):
        received.append((conversation_id, [row["content"] for row in messages]))
        return "reply " + messages[-1]["content"]
    monkeypatch.setattr(handler, "_respond", respond)
    with studio_project_session():
        handler([{"role": "user", "content": "A chemistry"}], conversation_id="A")
        handler([{"role": "user", "content": "B question"}], conversation_id="B")
        handler([{"role": "user", "content": "A followup"}], conversation_id="A")
    assert received[0][1] == ["A chemistry"]
    assert received[1][1] == ["B question"]
    assert received[2][1] == ["A chemistry", "reply A chemistry", "A followup"]
    assert handler.state_path == tmp_path / "state.json"


def test_checkpointed_thread_context_is_available_to_fresh_handler(tmp_path, monkeypatch):
    handler = RunWorkflowChatHandler({"state_path": str(tmp_path / "state.json")})
    seen = []
    monkeypatch.setattr(handler, "_respond", lambda messages, **kwargs: seen.extend(messages) or "ok")
    history = [{"role": "user", "content": "saved question"},
               {"role": "assistant", "content": "saved answer"},
               {"role": "user", "content": "continue"}]
    with studio_project_session(history), studio_project_session():
        handler([history[-1]], conversation_id="restored")
    assert seen == history

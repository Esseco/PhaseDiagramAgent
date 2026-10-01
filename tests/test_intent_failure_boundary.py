from unittest.mock import Mock
from decision_layer.agent.resolve_chat_intent import resolve_chat_intent
from run.open_webui_api import RunWorkflowChatHandler


def test_service_failure_does_not_expose_error_text():
    result = resolve_chat_intent("请看看", {},
        agent_client=Mock(side_effect=RuntimeError("secret response")))
    assert result["intent"] == "unavailable"
    assert result["error_type"] == "RuntimeError"
    assert "secret" not in str(result)


def test_invalid_response_is_not_an_other_intent():
    assert resolve_chat_intent("看看", {}, agent_client=lambda payload: []) == {
        "intent": "unavailable", "reason": "invalid_intent_response"}


def test_chat_does_not_advance_workflow_on_intent_failure(tmp_path):
    workflow = Mock()
    handler = RunWorkflowChatHandler({"state_path": str(tmp_path / "state.json"),
        "agent_client": Mock(side_effect=RuntimeError("secret response"))}, workflow=workflow)
    reply = handler([{"role": "user", "content": "帮我瞧瞧进度怎么样了"}])
    assert "意图识别暂时失败" in reply
    assert "secret" not in reply
    workflow.assert_not_called()

import json
from unittest.mock import Mock
from phase_agent.runtime.chat_application import RunWorkflowChatHandler
from phase_agent.runtime.conversation_context import conversation_context


def test_semantic_question_returns_short_answer_without_execution(tmp_path):
    from tests.test_unified_react_dialogue import make_handler
    from phase_agent.tools.step_runner.file_protocol import read_json
    client = Mock(return_value={"kind": "answer", "answer": "K折用于评估泛化误差。"})
    handler = make_handler(tmp_path, client)
    reply = handler([{"role": "user", "content": "这个K折究竟是干嘛的？"}])
    assert "K折用于" in reply
    assert client.call_count == 1
    assert not read_json(handler.state_path, {}).get("pending_execution_policies")






def test_context_is_bounded_allowlist_and_recent_dialogue():
    state = {"confirmed_config": {"api_key": "secret"}, "structures": [{"positions": [1, 2]}],
        "active_model_version": "m1", "remote_finetune_jobs": {"j": {
            "directory": "registered/training", "raw_data": "private", "status": "inputs_prepared"}}}
    snapshot = conversation_context(state, [{"role": "assistant", "content": "旧输入已改变"}])
    assert snapshot["recent_conversation"][0]["content"] == "旧输入已改变"
    assert snapshot["training_artifacts"][0]["directory"] == "registered/training"
    assert "secret" not in json.dumps(snapshot)
    assert "private" not in json.dumps(snapshot)





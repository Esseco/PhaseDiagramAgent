import json
from unittest.mock import Mock
from run.chat_application import RunWorkflowChatHandler
from run.chat_intent_routing import normalize_chat_intent
from run.conversation_context import conversation_context


def test_semantic_question_returns_short_answer_without_workflow(tmp_path):
    path = tmp_path / "state.json"
    state = {"active_model_version": "m1", "pending_execution_policies": {"p": {
        "agent_proposal": {"recommended_action": "update_mlip"}}}}
    path.write_text(json.dumps(state), encoding="utf-8")
    before = path.read_bytes()
    client = Mock(return_value={"intent": "read_only", "answer": "K折用于评估泛化误差，最终模型使用全部数据训练。"})
    workflow = Mock()
    handler = RunWorkflowChatHandler({"state_path": str(path), "agent_client": client}, workflow=workflow)
    reply = handler([{"role": "user", "content": "这个K折究竟是干嘛的？"}])
    assert "K折用于" in reply
    assert "本轮总结" not in reply
    assert "回复“同意”" not in reply
    workflow.assert_not_called()
    assert path.read_bytes() == before


def test_other_is_conversation_not_implicit_workflow():
    def client(payload):
        if payload["mode"] == "resolve_chat_intent":
            return {"intent": "other"}
        return {"kind": "answer", "answer": "这是旧输入与当前参数不一致，尚未覆盖。"}
    result = normalize_chat_intent("这句话什么意思", {}, agent_client=client, enable_context=True)
    assert result == {"reply": "这是旧输入与当前参数不一致，尚未覆盖。"}


def test_feedback_requires_explicit_request_not_model_approval():
    def result(response):
        return normalize_chat_intent("把方案改成四个成员", {}, enable_context=True, agent_client=lambda payload:
            {"intent": "other"} if payload["mode"] == "resolve_chat_intent" else response)
    assert "reply" in result({"kind": "approve", "direct_request": True, "confidence": 1})
    assert "reply" in result({"kind": "workflow_feedback", "direct_request": False, "confidence": 1})
    assert result({"kind": "workflow_feedback", "direct_request": True, "confidence": .98}) == {
        "message": "把方案改成四个成员"}


def test_context_is_bounded_allowlist_and_recent_dialogue():
    state = {"confirmed_config": {"api_key": "secret"}, "structures": [{"positions": [1, 2]}],
        "active_model_version": "m1", "remote_finetune_jobs": {"j": {
            "directory": "registered/training", "raw_data": "private", "status": "inputs_prepared"}}}
    snapshot = conversation_context(state, [{"role": "assistant", "content": "旧输入已改变"}])
    assert snapshot["recent_conversation"][0]["content"] == "旧输入已改变"
    assert snapshot["training_artifacts"][0]["directory"] == "registered/training"
    assert "secret" not in json.dumps(snapshot)
    assert "private" not in json.dumps(snapshot)


def test_conversation_failure_stays_read_only():
    client = Mock(side_effect=[{"intent": "other"}, RuntimeError("secret")])
    reply = normalize_chat_intent("解释一下", {}, agent_client=client, enable_context=True)
    assert "未推进任务" in reply["reply"]
    assert "secret" not in reply["reply"]


def test_context_external_transfer_is_disabled_by_default():
    client = Mock(return_value={"intent": "other"})
    result = normalize_chat_intent("解释一下", {"active_model_version": "private"}, agent_client=client,
                                  messages=[{"role": "assistant", "content": "private path"}])
    assert "reply" in result
    assert client.call_count == 1
    assert client.call_args.args[0]["context"]["conversation"] == {}

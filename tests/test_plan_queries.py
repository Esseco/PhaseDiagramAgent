import json
from unittest.mock import Mock
import pytest
from run.chat_application import RunWorkflowChatHandler
from run.plan_queries import pending_plan_reply, is_plan_query
from run.chat_intent_routing import normalize_chat_intent


@pytest.mark.parametrize("message", ["看看方案", "查看当前方案", "方案呢？", "看一下方案"])
def test_view_existing_plan_never_dispatches(tmp_path, message):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"pending_execution_policies": {"p": {"agent_proposal": {
        "recommended_action": "update_mlip", "expected_purpose": "改善受力误差"}}}}), encoding="utf-8")
    before = path.read_bytes()
    client, workflow = Mock(), Mock()
    handler = RunWorkflowChatHandler({"state_path": str(path), "agent_client": client},
                                     workflow=workflow, history_prompt=True)
    reply = handler([{"role": "user", "content": message}])
    assert "生成超算微调训练输入" in reply
    assert "改善受力误差" in reply
    assert "本轮总结" not in reply
    assert "请明确是询问" not in reply
    client.assert_not_called()
    workflow.assert_not_called()
    assert path.read_bytes() == before


def test_semantic_plan_view_uses_local_saved_plan():
    result = normalize_chat_intent("把刚才那份提议拿出来让我瞧瞧", {},
        agent_client=lambda payload: {"intent": "view_plan"})
    assert "没有待确认方案" in result["reply"]


def test_multiple_plans_do_not_choose_or_execute():
    reply = pending_plan_reply({"pending_execution_policies": {"a": {}, "b": {}}})
    assert "多个" in reply and "a" in reply and "b" in reply


def test_view_is_not_approval_or_regeneration():
    assert not is_plan_query("同意")
    assert not is_plan_query("重新生成方案")
    assert not is_plan_query("看看方案然后同意执行")

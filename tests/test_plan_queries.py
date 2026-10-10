import json
from unittest.mock import Mock
import pytest
from phase_agent.runtime.chat_application import RunWorkflowChatHandler
from phase_agent.runtime.plan_queries import pending_plan_reply, is_plan_query


@pytest.mark.parametrize("message", ["看看方案", "查看当前方案", "方案呢？", "看一下方案"])
def test_view_existing_plan_never_dispatches(tmp_path, message):
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            {
                "pending_execution_policies": {
                    "p": {
                        "agent_proposal": {
                            "recommended_action": "update_mlip",
                            "expected_purpose": "改善受力误差",
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    before = path.read_bytes()
    client, workflow = Mock(), Mock()
    handler = RunWorkflowChatHandler(
        {"state_path": str(path), "agent_client": client}, workflow=workflow, history_prompt=True
    )
    reply = handler([{"role": "user", "content": message}])
    assert "生成超算微调训练输入" in reply
    assert "改善受力误差" in reply
    assert "本轮总结" not in reply
    assert "请明确是询问" not in reply
    client.assert_not_called()
    workflow.assert_not_called()
    assert path.read_bytes() == before


def test_multiple_plans_do_not_choose_or_execute():
    reply = pending_plan_reply({"pending_execution_policies": {"a": {}, "b": {}}})
    assert "多个" in reply and "a" in reply and "b" in reply


def test_view_is_not_approval_or_regeneration():
    assert not is_plan_query("同意")
    assert not is_plan_query("重新生成方案")
    assert not is_plan_query("看看方案然后同意执行")


def test_blocked_branch_plan_shows_scope_without_inviting_approval():
    from phase_agent.runtime.studio_reply_presentation import format_turn_reply

    proposal = {
        "recommended_action": "generate_branches",
        "expected_purpose": "建立结构池",
        "action_parameters": {
            "quotas": {"coverage": 24},
            "max_det_H": 12,
            "batch_size": 8,
            "initial_states_per_branch": 3,
        },
        "raw_action": {"tool": "generate_branches", "task_key": "initial"},
    }
    state = {
        "pending_execution_policies": {"p": {"agent_proposal": proposal}},
        "effective_decisions": {"initial": {"status": "reserved"}},
    }
    before = json.dumps(state, sort_keys=True)
    reply = format_turn_reply("本轮未执行：相同任务已经提交或完成", state)
    assert "已保存方案" in reply and "24" in reply and "12" in reply
    assert "暂不可批准" in reply and "reserved" in reply
    assert "回复‘同意’批准" not in reply
    assert json.dumps(state, sort_keys=True) == before

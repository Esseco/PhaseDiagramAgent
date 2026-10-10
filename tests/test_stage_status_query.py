import pytest
from phase_agent.runtime.chat_application import _is_status_command
from phase_agent.runtime.status_presentation import format_progress


@pytest.mark.parametrize("message", ["现在什么阶段", "现在是什么阶段？", "目前处于什么阶段", "进行到什么阶段了"])
def test_stage_question_is_read_only_status(message):
    assert _is_status_command(message)


def test_stage_reply_uses_saved_agent_judgment():
    state = {"active_model_version": "old", "candidate_models": {"new": {
        "agent_review": {"choice": "other", "reason": "先核查受力接口"}}},
        "remote_finetune_jobs": {"j": {"status": "results_received", "directory": ".",
            "training_handoff": {"stage": "awaiting_direction_approval", "candidate_model_version": "new"}}}}
    result = format_progress(state)
    assert "当前阶段：微调后策略评估" in result
    assert "Agent建议：先核查受力接口" in result
    assert "回复‘同意’" in result

from execution_layer.workflows.run_tool_step import run_tool_step, _model_failure_action
from unittest.mock import patch
from unittest.mock import Mock
import json
from run.open_webui_api import RunWorkflowChatHandler, format_workflow_reply


def test_history_resume_refreshes_failed_proposal(tmp_path):
    proposal = {"recommended_action": "pause_search", "raw_action": {
        "tool": "pause_search", "decision_source": "rule", "fallback_reason": "llm_failed: length"}}
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"tasks": [{"task_id": "t", "status": "completed"}],
        "pending_execution_policies": {"p": {"agent_proposal": proposal}}}))
    handler = RunWorkflowChatHandler({"state_path": str(path)})
    handler._run = Mock(return_value={"status": "not_configured", "reason": "模型未返回有效方案"})
    handler([{"role": "user", "content": "继续"}])
    assert handler._run.call_count == 1
    assert handler._run.call_args.args[:2] == ("p", None)


def test_failed_proposal_has_no_approval_invitation():
    proposal = {"raw_action": {"decision_source": "rule", "fallback_reason": "llm_failed: length"}}
    text = format_workflow_reply({"status": "awaiting_approval", "agent_proposal": proposal}, "state.json")
    assert "不是暂停决策" in text
    assert "同意" not in text


def test_continue_discards_failed_pause_without_executing():
    action = {"tool": "pause_search", "decision_source": "rule",
              "fallback_reason": "llm_failed: truncated"}
    state = {"pending_execution_policies": {
        "__single_interactive_action__": {"agent_proposal": {"raw_action": action}}}}
    with patch("decision_layer.agent.choose_debug_next_action.choose_debug_next_action", return_value=None), \
         patch("execution_layer.workflows.run_tool_step.propose_agent_tool_action", return_value=action) as propose:
        result = run_tool_step(state, {"config": {}}, registry={},
                               execution_mode="interactive", context={"user_message": "继续"})
    assert propose.call_count == 1
    assert result["status"] == "not_configured"
    assert not result["state"]["pending_execution_policies"]
    assert "未暂停" in result["reason"]
    assert state["pending_execution_policies"]  # Input state is not mutated.


def test_model_failure_old_proposal_cannot_be_approved():
    action = {"tool": "pause_search", "decision_source": "rule",
              "fallback_reason": "llm_failed: truncated"}
    state = {"pending_execution_policies": {
        "__single_interactive_action__": {"agent_proposal": {"raw_action": action}}}}
    result = run_tool_step(state, {"config": {}}, registry={},
                           execution_mode="interactive",
                           human_feedback={"decision": "approve", "comment": "同意"})
    assert result["status"] == "rejected"
    assert "通信失败" in result["reason"]


def test_explicit_scientific_pause_is_not_model_failure():
    assert not _model_failure_action({"tool": "pause_search", "decision_source": "llm_agent"})
    assert not _model_failure_action({"tool": "pause_search", "decision_source": "rule",
                                     "fallback_reason": "llm_not_configured"})

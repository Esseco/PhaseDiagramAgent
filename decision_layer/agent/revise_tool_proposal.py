"""让 Agent 根据人工文字意见修订 action proposal，但不执行。"""

from copy import deepcopy
from config_layer.schema.action_state_schema import normalize_action


def revise_tool_proposal(proposal, comment, *, state, allowed_tools, agent_client):
    if agent_client is None:
        return {"action": deepcopy(proposal["raw_action"]), "analysis": "未配置 Agent，保留原 proposal。", "revision_status": "agent_not_configured"}
    payload = {
        "mode": "revise_action_from_human_feedback", "state": deepcopy(state),
        "allowed_tools": list(allowed_tools), "current_proposal": deepcopy(proposal),
        "human_comment": comment,
        "decision_context": deepcopy(state.get("decision_context") or {}),
        "instruction": "Analyze the human comment and return a revised action with action_type, parameters, reason, expected_cost, target_ids, expected_purpose and current_state_analysis. Do not execute it.",
    }
    try:
        action = agent_client(payload)
        if not isinstance(action, dict):
            raise TypeError("revised action must be a dict")
        action = normalize_action({**action, "decision_source": "llm_agent+human_feedback", "decision_context": payload["decision_context"]})
        return {"action": action, "analysis": action.get("current_state_analysis") or action.get("analysis"), "revision_status": "revised"}
    except Exception as error:
        return {"action": deepcopy(proposal["raw_action"]), "analysis": f"Agent 修订失败，保留原 proposal：{type(error).__name__}: {error}", "revision_status": "revision_failed"}

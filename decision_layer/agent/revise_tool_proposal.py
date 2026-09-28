"""让 Agent 根据人工文字意见修订 action proposal，但不执行。"""

import hashlib
import json
import re
from copy import deepcopy
from config_layer.schema.action_state_schema import normalize_action
from decision_layer.agent.resolve_explicit_generation_request import (
    _requested_branch_batch_size, _requested_max_det_H,
)


def revise_tool_proposal(proposal, comment, *, state, allowed_tools, agent_client):
    original = proposal.get("raw_action") or {}
    requested_limit = _requested_max_det_H(comment)
    requested_batch_size = _requested_branch_batch_size(comment)
    if ((requested_limit is not None or requested_batch_size is not None)
            and original.get("tool") == "generate_branches"
            and "generate_branches" in allowed_tools):
        action = deepcopy(original)
        action.pop("_llm_usage", None)
        params = action.setdefault("parameters", {})
        if requested_limit is not None:
            params["max_det_H"] = requested_limit
        if requested_batch_size is not None:
            params["batch_size"] = requested_batch_size
        initial_count = re.search(r"(?:取|初态(?:数)?(?:改为|为|设为)?)(\d+)个?初态?", re.sub(r"\s+", "", comment))
        if initial_count is None:
            initial_count = re.search(r"取(\d+)个初态", re.sub(r"\s+", "", comment))
        if initial_count:
            params["initial_states_per_branch"] = int(initial_count.group(1))
        for key in ("h_upper_bound", "max_H", "composition_constraints"):
            params.pop(key, None)
        identity = json.dumps([original.get("task_key"), requested_limit, requested_batch_size, comment],
                              ensure_ascii=False, default=str)
        action["task_key"] = "generate_branches:revised:" + hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()[:20]
        changes = []
        if requested_limit is not None:
            changes.append(f"det(H) ≤ {requested_limit}")
        if requested_batch_size is not None:
            changes.append(f"入选上限 {requested_batch_size} 个 branch")
        action["reason"] = "按人工反馈设置本轮" + "、".join(changes) + "。"
        action["expected_purpose"] = action.get("expected_purpose") or "按已批准的 H 上限生成 branch。"
        action["decision_source"] = "explicit_user_instruction"
        return {"action": normalize_action(action), "analysis": action["reason"],
                "revision_status": "user_generation_limits_applied"}
    if agent_client is None:
        retained = deepcopy(proposal["raw_action"])
        retained.pop("_llm_usage", None)
        return {"action": retained, "analysis": "未配置 Agent，保留原 proposal。", "revision_status": "agent_not_configured"}
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
        tool = action.get("tool") or action.get("action_type")
        if not isinstance(tool, str) or tool not in allowed_tools:
            raise ValueError(f"修订动作无效：{tool!r}")
        if tool == "generate_branches":
            params = action.setdefault("parameters", {})
            requested_limit = _requested_max_det_H(comment)
            if requested_limit is not None:
                params["max_det_H"] = requested_limit
            requested_batch_size = _requested_branch_batch_size(comment)
            if requested_batch_size is not None:
                params["batch_size"] = requested_batch_size
            aliases = {key: params[key] for key in ("h_upper_bound", "max_H") if key in params}
            if aliases and requested_limit is None:
                raise ValueError(f"生成器不识别这些 H 上限字段：{sorted(aliases)}；请使用 max_det_H")
            for key in aliases:
                del params[key]
            params.pop("composition_constraints", None)
        if tool not in {"check_convergence", "pause_search"} and not action.get("task_key"):
            identity = {
                "original_task_key": (proposal.get("raw_action") or {}).get("task_key"),
                "comment": comment, "tool": tool,
                "parameters": action.get("parameters") or {},
                "target_ids": action.get("target_ids") or [],
            }
            digest = hashlib.sha256(json.dumps(
                identity, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()[:20]
            action["task_key"] = f"agent-revision:{digest}"
        action = normalize_action({**action, "decision_source": "llm_agent+human_feedback", "decision_context": payload["decision_context"]})
        return {"action": action, "analysis": action.get("current_state_analysis") or action.get("analysis"), "revision_status": "revised"}
    except Exception as error:
        original_action = deepcopy(proposal["raw_action"])
        original_action.pop("_llm_usage", None)
        return {"action": original_action, "llm_usage": getattr(error, "llm_usage", None),
                "analysis": f"Agent 修订失败，保留原 proposal：{type(error).__name__}: {error}", "revision_status": "revision_failed"}

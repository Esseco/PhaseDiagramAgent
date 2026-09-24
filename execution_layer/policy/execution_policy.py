"""Execution gating for Agent-proposed tool actions.

This module only decides whether an action may continue to validation. It does
not validate actions, reserve budget, invoke tools, or interpret scientific
results.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any
from analysis_layer.cost.estimate_proposal_cost import estimate_proposal_cost
from config_layer.schema.action_state_schema import normalize_action


EXECUTION_MODES = {"interactive", "autonomous", "dry_run", "replay"}
HUMAN_DECISIONS = {"approve", "modify", "reject"}


def build_agent_proposal(action: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """Create the stable proposal envelope shown and stored by the policy."""
    action = normalize_action(action)
    analysis = action.get("analysis") or action.get("current_state_analysis")
    if analysis is None:
        analysis = {
            "status": state.get("status"),
            "active_tasks": len(state.get("tasks") or state.get("pending_tasks") or []),
            "decision_source": action.get("decision_source"),
        }
    cost_plan = estimate_proposal_cost(action, state)
    return {
        "current_state_analysis": deepcopy(analysis),
        "recommended_action": action.get("action_type"),
        "action_parameters": deepcopy(action.get("parameters") or {}),
        "reason": action.get("reason"),
        "estimated_cost": cost_plan,
        "calculation_plan": deepcopy(cost_plan["workload"]),
        "expected_purpose": action.get("expected_purpose") or action.get("purpose") or action.get("reason"),
        "raw_action": deepcopy(action),
        "decision_context": deepcopy(action.get("decision_context") or {}),
        "evidence_refs": deepcopy(action.get("evidence_refs") or []),
    }


def apply_execution_policy(
    agent_proposal: dict[str, Any],
    *,
    execution_mode: str,
    human_feedback: str | dict[str, Any] | None = None,
    replay_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Gate a proposal before validation and return the proposed final action."""
    if execution_mode not in EXECUTION_MODES:
        raise ValueError(f"未知 execution_mode：{execution_mode}")
    original = deepcopy(agent_proposal["raw_action"])
    if execution_mode == "autonomous":
        return {"status": "approved", "decision": "approve", "human_feedback": None, "final_action": original, "execute": True}
    if execution_mode == "dry_run":
        return {"status": "dry_run", "decision": "approve", "human_feedback": None, "final_action": original, "execute": False}
    if execution_mode == "replay":
        if not replay_record or not isinstance(replay_record.get("final_action"), dict):
            raise ValueError("replay 模式需要包含 final_action 的 replay_record")
        return {"status": "approved", "decision": "replay", "human_feedback": None, "final_action": deepcopy(replay_record["final_action"]), "execute": True}
    feedback = _normalize_human_feedback(human_feedback)
    if feedback is None:
        return {"status": "awaiting_approval", "decision": None, "human_feedback": None, "final_action": None, "execute": False}
    decision = feedback["decision"]
    if decision == "reject":
        return {"status": "rejected_by_user", "decision": decision, "human_feedback": feedback, "final_action": None, "execute": False}
    final_action = original if decision == "approve" else _apply_modifications(original, feedback["modifications"])
    return {"status": "approved", "decision": decision, "human_feedback": feedback, "final_action": final_action, "execute": True}


def _normalize_human_feedback(feedback):
    if feedback is None:
        return None
    if isinstance(feedback, str):
        feedback = {"decision": "approve", "comment": feedback} if feedback.lower() in {"approve", "同意"} else {"decision": feedback}
    if not isinstance(feedback, dict) or feedback.get("decision") not in HUMAN_DECISIONS:
        if isinstance(feedback, dict) and str(feedback.get("comment") or "").strip().lower() in {"approve", "同意"}:
            feedback = {**feedback, "decision": "approve"}
        else:
            raise ValueError("human_feedback.decision 必须是 approve/modify/reject")
    normalized = deepcopy(feedback)
    if normalized["decision"] in {"approve", "modify"} and not _has_explicit_approval(normalized.get("comment")):
        raise ValueError("interactive 执行要求 comment 最后一条非空内容为‘同意’或 approve")
    modifications = normalized.get("modifications") or normalized.get("action") or {}
    if normalized["decision"] == "modify" and not isinstance(modifications, dict):
        raise ValueError("modify 需要字典类型 modifications")
    normalized["modifications"] = deepcopy(modifications)
    return normalized


def _has_explicit_approval(comment):
    if not isinstance(comment, str):
        return False
    lines = [line.strip() for line in comment.splitlines() if line.strip()]
    return bool(lines) and lines[-1].lower() in {"approve", "同意"}


def _apply_modifications(action: dict[str, Any], modifications: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(action)
    for key, value in modifications.items():
        if key == "parameters" and isinstance(value, dict):
            updated["parameters"] = _deep_merge(updated.get("parameters") or {}, value)
        else:
            updated[key] = deepcopy(value)
    updated["decision_source"] = f"{updated.get('decision_source', 'agent')}+human_modified"
    return updated


def _deep_merge(original: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(original)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged

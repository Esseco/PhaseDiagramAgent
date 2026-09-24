"""在执行前校验动作、预算、目标、权重和重复任务。"""

from execution_layer.budget.check_budget import check_budget


def validate_agent_action(action: dict, summary: dict, *, config=None) -> dict:
    settings = {"weight_bounds": [0.0, 1.0], "allowed_strategies": ["coverage", "composition", "competing_phase", "tm_ordering", "periodic_extension", "random_exploration"], "minimum_exploration_fraction": 0.05, "minimum_dft_audit_fraction": 0.05}
    settings.update(config or {})
    errors = []
    if action.get("action") not in summary.get("allowed_actions", []):
        errors.append("action_not_allowed")
    budget = action.get("budget", 0.0)
    remaining = summary.get("remaining_budget")
    if not isinstance(budget, (int, float)) or budget < 0 or (remaining is not None and budget > remaining):
        errors.append("invalid_or_excess_budget")
    known = set(summary.get("known_target_ids", [])) | set(summary.get("candidate_ids", []))
    if any(target not in known for target in action.get("target_ids", [])):
        errors.append("unknown_target")
    if action.get("strategy") and action["strategy"] not in settings["allowed_strategies"]:
        errors.append("strategy_not_allowed")
    low, high = settings["weight_bounds"]
    if any(not isinstance(value, (int, float)) or not low <= value <= high for value in action.get("weights", {}).values()):
        errors.append("weight_out_of_bounds")
    quotas = action.get("quotas", {})
    if any(not isinstance(value, int) or value < 0 for value in quotas.values()):
        errors.append("invalid_quota")
    if quotas and sum(quotas.values()) > int(action.get("quota_budget", sum(quotas.values()))):
        errors.append("quota_budget_exceeded")
    if action.get("exploration_fraction", settings["minimum_exploration_fraction"]) < settings["minimum_exploration_fraction"]:
        errors.append("exploration_floor_violated")
    if action.get("dft_audit_fraction", settings["minimum_dft_audit_fraction"]) < settings["minimum_dft_audit_fraction"]:
        errors.append("dft_audit_floor_violated")
    task_key = action.get("task_key")
    if task_key and task_key in set(summary.get("active_task_keys", [])) | set(summary.get("completed_task_keys", [])) and not action.get("allow_repeat"):
        errors.append("duplicate_task")
    forbidden = {"energy", "ehull", "reward", "confidence"} & set(action)
    if forbidden:
        errors.append("agent_supplied_numeric_result")
    if settings.get("budget_limits"):
        budget_check = check_budget(summary, {"stage": action.get("stage"), "tasks": len(action.get("target_ids", [])) or 1, "relative_cost": action.get("budget", 0.0)}, settings["budget_limits"])
        errors.extend(budget_check["reasons"])
    return {"valid": not errors, "errors": errors, "checked_action": action, "basis": settings}

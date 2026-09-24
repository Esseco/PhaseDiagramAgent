"""Validate an explicit, expiring authorization snapshot for automatic mode."""

from datetime import datetime, timezone

SENSITIVE_ACTIONS = {"dft_relax", "update_mlip", "activate_model", "change_hard_constraint"}


def authorize_automatic_action(action, state, authorization, *, now=None):
    now = now or datetime.now(timezone.utc)
    if not authorization or authorization.get("enabled") is not True:
        return {"allowed": False, "reason": "automatic_mode_not_explicitly_enabled"}
    try:
        expires = datetime.fromisoformat(str(authorization["expires_at"]).replace("Z", "+00:00"))
    except (KeyError, ValueError):
        return {"allowed": False, "reason": "authorization_expiry_invalid"}
    if now >= expires:
        return {"allowed": False, "reason": "authorization_expired"}
    tool = action.get("tool") or action.get("stage")
    if tool in SENSITIVE_ACTIONS:
        return {"allowed": False, "reason": "sensitive_action_requires_separate_confirmation"}
    if tool not in set(authorization.get("allowed_actions") or []):
        return {"allowed": False, "reason": "action_outside_authorized_set"}
    cost = float(action.get("budget", action.get("planned_relative_cost", 0)) or 0)
    if cost > float(authorization.get("max_batch_cost", 0)):
        return {"allowed": False, "reason": "batch_cost_limit"}
    used = float((state.get("automatic_mode_usage") or {}).get("cost", 0) or 0)
    if used + cost > float(authorization.get("total_budget", 0)):
        return {"allowed": False, "reason": "automatic_total_budget"}
    active = sum(row.get("status") in {"pending", "running", "submitted"} for row in state.get("tasks") or [])
    if active >= int(authorization.get("concurrency_limit", 0)):
        return {"allowed": False, "reason": "automatic_concurrency_limit"}
    return {"allowed": True, "reason": "within_explicit_authorization", "cost": cost}

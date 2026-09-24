"""Reserve budget idempotently before dispatching a formal task."""

from copy import deepcopy


def reserve_action_budget(state: dict, action: dict, *, config_version: str) -> dict:
    updated = deepcopy(state); key = action.get("task_key")
    if not key:
        raise ValueError("formal action requires task_key")
    existing = updated.setdefault("effective_decisions", {}).get(key)
    if existing:
        return updated
    cost = float(action.get("budget", 0)); reservations = updated.setdefault("budget_reservations", {})
    reservations[key] = {"relative_cost": cost, "reserved_cost": cost, "stage": action.get("stage") or action.get("tool"), "status": "reserved", "config_version": config_version}
    updated["reserved_relative_cost"] = float(updated.get("reserved_relative_cost", 0)) + cost
    updated["effective_decisions"][key] = {"status": "reserved", "tool": action["tool"], "config_version": config_version}
    return updated

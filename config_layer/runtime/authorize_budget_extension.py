"""Rebind a persisted run only when a user-confirmed revision increases budgets."""

from copy import deepcopy


def authorize_budget_extension(state, confirmed_snapshot, *, user_approved=False):
    current = deepcopy(state)
    new_version = confirmed_snapshot["config_version"]
    old_version = current.get("confirmed_config_version")
    if old_version in {None, new_version}:
        current["confirmed_config_version"] = new_version
        current.setdefault("confirmed_config", deepcopy(confirmed_snapshot["config"]))
        return {"status": "unchanged", "state": current}
    if not user_approved:
        return {"status": "approval_required", "state": current}
    old = current.get("confirmed_config")
    new = deepcopy(confirmed_snapshot["config"])
    if not old or _without_budgets(old) != _without_budgets(new):
        return {"status": "rejected_non_budget_change", "state": current}
    if not _non_decreasing(old.get("budgets") or {}, new.get("budgets") or {}):
        return {"status": "rejected_budget_decrease", "state": current}
    current["confirmed_config_version"] = new_version
    current["confirmed_config"] = new
    current.setdefault("config_migrations", []).append({"type": "approved_budget_extension", "from": old_version, "to": new_version})
    return {"status": "extended", "state": current}


def _without_budgets(config):
    value = deepcopy(config); value.pop("budgets", None); return value


def _non_decreasing(old, new):
    for key, value in old.items():
        replacement = new.get(key)
        if isinstance(value, dict):
            if not isinstance(replacement, dict) or not _non_decreasing(value, replacement): return False
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if replacement is not None and float(replacement) < float(value): return False
        elif replacement != value:
            return False
    return True

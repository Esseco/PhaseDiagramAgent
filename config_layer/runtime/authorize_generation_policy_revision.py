"""Rebind a confirmed run after safe generation and Relax-capacity changes."""

from copy import deepcopy


_ALLOWED = {
    ("run", "total_quota"), ("run", "batch_size"),
    ("run", "initial_states_per_branch"),
    ("run", "generation_options", "selection_config", "max_per_framework"),
    ("round_strategy", "generation_quota_total"),
    ("round_strategy", "rule_default", "generation_quotas"),
    ("round_strategy", "rule_default", "mc_budget"),
    ("round_strategy", "maximum_mc_budget"),
    ("budgets", "stage_limits", "relax_and_feature", "max_tasks"),
    ("budgets", "stage_limits", "relax_and_feature", "max_cost"),
}


def authorize_generation_policy_revision(state, confirmed_snapshot, *, allowed_paths=None):
    """Apply only confirmed, non-scientific scheduling changes to an idle run."""
    current = deepcopy(state)
    old_version = current.get("confirmed_config_version")
    new_version = confirmed_snapshot.get("config_version")
    if not old_version or old_version == new_version:
        return {"status": "unchanged", "state": current}
    old = current.get("confirmed_config") or {}
    new = deepcopy(confirmed_snapshot.get("config") or {})
    if not old or not _same_outside_allowed(old, new, allowed_paths=allowed_paths):
        return {"status": "rejected_scientific_change", "state": current}
    if not _no_active_work(current):
        return {"status": "active_work", "state": current}
    if not _non_decreasing_relax_limit(old, new):
        return {"status": "rejected_limit_decrease", "state": current}
    old_mc_cap = (old.get("round_strategy") or {}).get("maximum_mc_budget")
    new_mc_cap = (new.get("round_strategy") or {}).get("maximum_mc_budget")
    if old_mc_cap is not None and (new_mc_cap is None or new_mc_cap < old_mc_cap):
        return {"status": "rejected_limit_decrease", "state": current}
    new_mc_default = ((new.get("round_strategy") or {}).get("rule_default") or {}).get("mc_budget")
    if (new_mc_default is not None and new_mc_cap is not None
            and new_mc_default > new_mc_cap):
        return {"status": "rejected_limit_decrease", "state": current}
    current["confirmed_config_version"] = new_version
    current["confirmed_config"] = new
    current.setdefault("config_migrations", []).append({
        "type": "confirmed_generation_policy_revision",
        "from": old_version, "to": new_version,
    })
    return {"status": "rebound", "state": current}


def authorize_second_mc_segment_revision(state, confirmed_snapshot):
    """Bind only an explicitly confirmed false-to-true second MC segment change."""
    old = (state.get("confirmed_config") or {})
    new = (confirmed_snapshot.get("config") or {})
    path = ("mc_policy", "second_segment_enabled")
    if (((old.get("mc_policy") or {}).get("second_segment_enabled") is not False)
            or ((new.get("mc_policy") or {}).get("second_segment_enabled") is not True)
            or not state.get("confirmed_config_version")
            or state.get("confirmed_config_version") == confirmed_snapshot.get("config_version")
            or not _same_outside_allowed(old, new, allowed_paths={path})):
        return {"status": "not_applicable", "state": deepcopy(state)}
    result = authorize_generation_policy_revision(
        state, confirmed_snapshot, allowed_paths={path})
    if result["status"] == "active_work":
        result["status"] = "rejected_active_work"
    if result["status"] == "rebound":
        result["changed_fields"] = ["mc_policy.second_segment_enabled"]
    return result


def _same_outside_allowed(old, new, *, allowed_paths=None):
    before, after = deepcopy(old), deepcopy(new)
    for tree in (before, after):
        for path in _ALLOWED if allowed_paths is None else allowed_paths:
            _remove(tree, path)
    return before == after


def _remove(tree, path):
    cursor = tree
    ancestors = []
    for part in path[:-1]:
        if not isinstance(cursor, dict) or part not in cursor:
            return
        ancestors.append((cursor, part))
        cursor = cursor[part]
    if isinstance(cursor, dict):
        cursor.pop(path[-1], None)
    for parent, key in reversed(ancestors):
        if parent.get(key) == {}:
            parent.pop(key, None)


def _no_active_work(state):
    active = {"pending", "planned", "reserved", "submitted", "running", "queued"}
    if any((row or {}).get("status") in active for row in state.get("tasks") or []):
        return False
    if any((row or {}).get("status") in active
           for row in (state.get("budget_reservations") or {}).values()):
        return False
    return not (state.get("pending_execution_policies") or {})


def _non_decreasing_relax_limit(old, new):
    before = (((old.get("budgets") or {}).get("stage_limits") or {})
              .get("relax_and_feature") or {})
    after = (((new.get("budgets") or {}).get("stage_limits") or {})
             .get("relax_and_feature") or {})
    for key in ("max_tasks", "max_cost"):
        left, right = before.get(key), after.get(key)
        if left is None:
            continue
        if right is None or float(right) < float(left):
            return False
    return True

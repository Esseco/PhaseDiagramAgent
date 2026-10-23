"""Rebind only the explicitly approved future DFT selection policy."""

from copy import deepcopy
import json


_POLICY = {
    "max_tasks_per_round": 100,
    "single_point_max_per_round": 100,
    "single_point_cost_per_round": None,
}


def authorize_dft_selection_revision(state, snapshot, *, user_approved=False):
    old_version = state.get("confirmed_config_version")
    new_version = snapshot.get("config_version")
    if not old_version or old_version == new_version:
        return None
    old = state.get("confirmed_config") or {}
    new = snapshot.get("config") or {}
    before = (old.get("dft") or {}).get("selection") or {}
    after = (new.get("dft") or {}).get("selection") or {}
    missing = object()
    changes = []
    for field in _POLICY:
        left, right = before.get(field, missing), after.get(field, missing)
        if left != right or type(left) is not type(right):
            changes.append(
                {
                    "field": "dft.selection." + field,
                    "from": None if left is missing else deepcopy(left),
                    "from_present": left is not missing,
                    "to": None if right is missing else deepcopy(right),
                }
            )
    if not changes:
        return None
    result = {"state": deepcopy(state), "from": old_version, "to": new_version}
    if user_approved is not True:
        return {**result, "status": "approval_required"}
    if (
        any(
            type(after.get(key)) is not int or after[key] != 100
            for key in ("max_tasks_per_round", "single_point_max_per_round")
        )
        or "single_point_cost_per_round" not in after
        or after["single_point_cost_per_round"] is not None
    ):
        return {**result, "status": "rejected_invalid_dft_selection_policy"}
    expected = deepcopy(old)
    expected.setdefault("dft", {}).setdefault("selection", {}).update(_POLICY)
    if json.dumps(expected, sort_keys=True, default=str) != json.dumps(
        new, sort_keys=True, default=str
    ):
        from phase_agent.configuration.runtime.authorize_budget_extension import _changed_paths

        return {
            **result,
            "status": "rejected_non_budget_change",
            "changed_fields": _changed_paths(old, new),
        }
    current = result["state"]
    current["confirmed_config_version"] = new_version
    current["confirmed_config"] = deepcopy(new)
    current.setdefault("config_migrations", []).append(
        {
            "type": "approved_dft_selection_policy_revision",
            "from": old_version,
            "to": new_version,
            "policy_changes": deepcopy(changes),
        }
    )
    return {**result, "status": "extended", "policy_changes": changes}

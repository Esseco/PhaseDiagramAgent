"""Per-batch Relax–MC branch limit and reservation policy."""

from copy import deepcopy

MAX_RELAX_MC_BRANCHES = 500


def validate_branch_scope(branch_ids):
    if (
        not isinstance(branch_ids, (list, tuple))
        or not branch_ids
        or any(not isinstance(x, str) or not x for x in branch_ids)
        or len(set(branch_ids)) != len(branch_ids)
    ):
        raise ValueError("relax_mc_branch_scope_invalid")
    if len(branch_ids) > MAX_RELAX_MC_BRANCHES:
        raise ValueError("relax_mc_branch_limit_exceeded")


def batch_reservation_limits(limits, branch_ids):
    """The approved batch limits branches, not lifetime MC task history.

    Other stages, resource rules, actual costs and the recorded cumulative
    accounting remain intact; this does not authorize or submit any task.
    """
    validate_branch_scope(branch_ids)
    result = deepcopy(limits)
    deep = (result.get("stage_limits") or {}).get("deep_search")
    if not isinstance(deep, dict):
        raise ValueError("relax_mc_deep_search_limits_missing")
    deep["max_tasks"] = None
    return result

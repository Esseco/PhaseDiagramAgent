"""Prevent Agent MC allocation changes while a frozen BOHB round owns tasks."""


def check_bohb_task_ownership(round_state: dict | None, action: dict) -> dict:
    if not round_state or round_state.get("status") not in {"active", "paused"}:
        return {"allowed": True, "reason": None}
    branch_ids = set(round_state.get("candidate_ids") or [])
    targets = set(
        action.get("target_ids") or ([action.get("branch_id")] if action.get("branch_id") else [])
    )
    modifies_mc = (
        action.get("tool") in {"run_calculation_stage", "allocate_mc", "change_mc_budget"}
        and action.get("stage", "deep_search") == "deep_search"
    )
    if modifies_mc and targets & branch_ids:
        return {
            "allowed": False,
            "reason": "active_bohb_round_owns_mc_allocation",
            "strategy_version": round_state.get("strategy_version"),
        }
    return {"allowed": True, "reason": None}

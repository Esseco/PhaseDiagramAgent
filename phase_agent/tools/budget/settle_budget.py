"""Settle actual consumed cost once and release unused reservation."""

from copy import deepcopy


def settle_budget(
    state: dict,
    *,
    task_key: str,
    settlement_id: str,
    task_status: str,
    actual_cost,
    estimated_cost=None,
    cost_source=None,
    failure_reason=None,
) -> dict:
    updated = deepcopy(state)
    reservation = updated.setdefault("budget_reservations", {}).get(task_key)
    if reservation is None:
        return {"status": "unknown_reservation", "state": updated}
    if reservation.get("settlement_id") == settlement_id or reservation.get("status") == "settled":
        return {"status": "already_settled", "state": updated, "reservation": reservation}
    if task_status not in {"completed", "failed", "timeout", "cancelled"}:
        return {"status": "not_terminal", "state": updated}
    reserved = float(reservation.get("reserved_cost", reservation.get("relative_cost", 0)))
    if actual_cost is not None:
        actual = max(0.0, float(actual_cost))
        estimated = None
        accounted = actual
        basis = cost_source or "reported_actual_cost"
    else:
        actual = None
        estimated = max(0.0, float(estimated_cost)) if estimated_cost is not None else reserved
        accounted = estimated
        basis = cost_source or (
            "estimated_cost"
            if estimated_cost is not None
            else "reserved_cost_upper_bound_actual_unknown"
        )
    released = max(0.0, reserved - accounted)
    usage = updated.setdefault("budget_usage", {})
    usage["total_relative_cost"] = float(usage.get("total_relative_cost", 0)) + accounted
    if actual is not None:
        usage["actual_total_relative_cost"] = (
            float(usage.get("actual_total_relative_cost", 0)) + actual
        )
    else:
        usage["estimated_total_relative_cost"] = (
            float(usage.get("estimated_total_relative_cost", 0)) + estimated
        )
    stage_name = reservation.get("stage", "unclassified")
    stage = usage.setdefault("stages", {}).setdefault(stage_name, {"tasks": 0, "cost": 0.0})
    stage["tasks"] += 1
    stage["cost"] += accounted
    key = "actual_cost" if actual is not None else "estimated_cost"
    stage[key] = float(stage.get(key, 0)) + (actual if actual is not None else estimated)
    updated["reserved_relative_cost"] = max(
        0.0, float(updated.get("reserved_relative_cost", 0)) - reserved
    )
    reservation.update(
        {
            "status": "settled",
            "task_status": task_status,
            "settlement_id": settlement_id,
            "actual_cost": actual,
            "estimated_cost": estimated,
            "accounted_cost": accounted,
            "actual_cost_known": actual is not None,
            "released_cost": released,
            "settlement_basis": basis,
            "failure_reason": failure_reason,
        }
    )
    return {"status": "settled", "state": updated, "reservation": reservation}

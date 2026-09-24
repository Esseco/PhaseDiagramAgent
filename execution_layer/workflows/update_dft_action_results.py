"""Reconcile DFT child tasks with the shared reservation/settlement ledger."""

from copy import deepcopy

from execution_layer.budget.settle_budget import settle_budget


def update_dft_action_results(state: dict, submissions: list[dict]) -> dict:
    updated = deepcopy(state)
    effective = updated.setdefault("effective_decisions", {})
    reservations = updated.setdefault("budget_reservations", {})
    for result in submissions:
        key = result.get("task_key")
        if key not in effective:
            continue
        status = result.get("status")
        terminal = status in {"completed", "failed", "cancelled", "timeout"}
        if effective[key].get("result_recorded") and terminal:
            continue
        effective[key]["status"] = status
        effective[key]["error"] = result.get("error")
        reservation = reservations.get(key)
        if reservation is not None and status in {"pending", "running"}:
            reservation["status"] = "submitted" if status == "pending" else "running"
        if not terminal:
            continue
        before_reserved = float(updated.get("reserved_dft_cost", 0))
        settlement = settle_budget(
            updated,
            task_key=key,
            settlement_id=result.get("settlement_id") or result.get("task_id") or key,
            task_status=status,
            actual_cost=result.get("actual_cost"),
            failure_reason=result.get("error"),
        )
        updated = settlement["state"]
        effective = updated.setdefault("effective_decisions", {})
        reservations = updated.setdefault("budget_reservations", {})
        settled = reservations.get(key) or {}
        effective[key].update({"status": status, "result_recorded": True, "error": result.get("error")})
        reserved_cost = float(settled.get("reserved_cost", settled.get("relative_cost", 0)))
        updated["reserved_dft_cost"] = max(0.0, before_reserved - reserved_cost)
        if status in {"completed", "failed", "timeout"} and not settled.get("dft_compatibility_charged"):
            updated["used_dft_cost"] = float(updated.get("used_dft_cost", 0)) + float(
                settled.get("accounted_cost", settled.get("actual_cost", 0)) or 0)
            settled["dft_compatibility_charged"] = True
    return updated

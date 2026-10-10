"""Settle timed-out submitted tasks or release never-submitted reservations."""

from datetime import datetime, timezone

from phase_agent.tools.budget.release_budget import release_budget
from phase_agent.tools.budget.settle_budget import settle_budget


def expire_budget_reservations(state: dict, *, now=None) -> dict:
    current = state
    timestamp = now or datetime.now(timezone.utc).isoformat()
    events = []
    for key, item in list((current.get("budget_reservations") or {}).items()):
        if (
            item.get("status") not in {"reserved", "submitted"}
            or not item.get("timeout_at")
            or item["timeout_at"] > timestamp
        ):
            continue
        if item["status"] == "submitted":
            result = settle_budget(
                current,
                task_key=key,
                settlement_id=f"timeout:{key}:{item['timeout_at']}",
                task_status="timeout",
                actual_cost=item.get("reported_cost"),
                failure_reason="timeout",
            )
        else:
            result = release_budget(
                current, task_key=key, reason="reservation_timeout_before_submission"
            )
        current = result["state"]
        events.append({"task_key": key, "result": result["status"]})
    return {"state": current, "events": events}

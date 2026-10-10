"""Release an unsubmitted or explicitly cancelled reservation."""

from copy import deepcopy


def release_budget(state: dict, *, task_key: str, reason: str) -> dict:
    updated = deepcopy(state)
    record = updated.setdefault("budget_reservations", {}).get(task_key)
    if record is None:
        return {"status": "unknown_reservation", "state": updated}
    if record.get("status") in {"released", "settled"}:
        return {"status": "unchanged", "state": updated, "reservation": record}
    record.update(
        {"status": "released", "released_cost": record["reserved_cost"], "release_reason": reason}
    )
    return {"status": "released", "state": updated, "reservation": record}

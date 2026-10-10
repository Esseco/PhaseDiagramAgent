"""Reserve budget before submission, idempotently by task key."""

from copy import deepcopy

from phase_agent.tools.budget.check_budget import check_budget


def reserve_budget(
    state: dict,
    *,
    task_key: str,
    stage: str,
    amount: float,
    limits: dict,
    config_version=None,
    model_version=None,
    reserved_at=None,
    timeout_at=None,
) -> dict:
    updated = deepcopy(state)
    reservations = updated.setdefault("budget_reservations", {})
    if task_key in reservations:
        return {"status": "existing", "state": updated, "reservation": reservations[task_key]}
    check = check_budget(updated, {"stage": stage, "tasks": 1, "relative_cost": amount}, limits)
    if not check["allowed"]:
        return {"status": "rejected", "state": updated, "reasons": check["reasons"]}
    record = {
        "task_key": task_key,
        "stage": stage,
        "reserved_cost": float(amount),
        "status": "reserved",
        "config_version": config_version,
        "model_version": model_version,
        "reserved_at": reserved_at,
        "timeout_at": timeout_at,
        "settlement_id": None,
        "actual_cost": None,
        "released_cost": None,
    }
    reservations[task_key] = record
    return {"status": "reserved", "state": updated, "reservation": record}

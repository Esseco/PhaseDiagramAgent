"""Settle DFT cancellations only when the recorded scheduler identity agrees."""

from copy import deepcopy

from phase_agent.tools.budget.settle_budget import _settle_budget_in_place


def settle_cancelled(state, task_ids):
    if (
        not isinstance(task_ids, list)
        or not task_ids
        or any(not isinstance(item, str) or not item for item in task_ids)
        or len(set(task_ids)) != len(task_ids)
    ):
        raise ValueError("cancelled_dft_scope_invalid")
    tasks = state.get("tasks") or []
    by_id = {row.get("task_id"): row for row in tasks}
    if len(by_id) != len(tasks):
        raise ValueError("duplicate_task_id")
    reservations = state.get("budget_reservations") or {}
    for task_id in task_ids:
        task = by_id.get(task_id) or {}
        evidence = task.get("scheduler_terminal_evidence") or {}
        key = task.get("task_key")
        reservation = reservations.get(key) or {}
        if (
            task.get("stage") != "dft_single_point"
            or task.get("status") != "cancelled"
            or task.get("recovery_cancelled") is not True
            or evidence.get("state") not in {"CANCELLED", "CANCELLED+"}
            or not evidence.get("job_id")
            or evidence.get("task_id") != task_id
            or evidence.get("task_key") != key
            or not task.get("task_checksum")
            or evidence.get("task_checksum") != task.get("task_checksum")
            or evidence.get("batch_id") != task.get("batch_id")
            or reservation.get("stage") != "dft_single_point"
            or reservation.get("status") not in {"reserved", "submitted", "running", "settled"}
        ):
            raise ValueError("cancelled_dft_terminal_evidence_invalid:" + task_id)
        if reservation.get("status") == "settled" and reservation.get("task_status") != "cancelled":
            raise ValueError("cancelled_dft_settlement_conflict:" + task_id)
    current = deepcopy(state)
    settled = []
    for task_id in task_ids:
        task = by_id[task_id]
        result = _settle_budget_in_place(
            current,
            task_key=task["task_key"],
            settlement_id=task_id,
            task_status="cancelled",
            actual_cost=task.get("actual_cost"),
            estimated_cost=task.get("estimated_cost"),
            cost_source=task.get("estimated_cost_basis"),
            failure_reason=task.get("cancellation_reason"),
        )
        if result["status"] == "settled":
            settled.append(task_id)
    return current, {
        "status": "settled" if settled else "already_settled",
        "task_ids": list(task_ids),
        "settled_count": len(settled),
        "submitted": False,
    }

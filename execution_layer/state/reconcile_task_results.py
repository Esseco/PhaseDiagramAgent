"""Reconcile asynchronous task results and settle reservations exactly once."""

from copy import deepcopy

from execution_layer.budget.settle_budget import settle_budget
from execution_layer.cost.extract_gpu_accounting import extract_gpu_accounting


TERMINAL_STATUSES = {"completed", "failed", "timeout", "cancelled"}
ACTIVE_STATUSES = {"pending", "running"}


def reconcile_task_results(state: dict | None, recovered_results=None) -> dict:
    current = deepcopy(state or {})
    tasks = current.setdefault("tasks", deepcopy(current.get("pending_tasks") or []))
    processed = current.setdefault("processed_task_ids", [])
    reconciled = []
    for incoming in recovered_results or []:
        result = deepcopy(incoming)
        task_id = result.get("task_id")
        task_key = result.get("task_key")
        status = result.get("status")
        if not task_id or not task_key or status not in ACTIVE_STATUSES | TERMINAL_STATUSES:
            reconciled.append({"status": "rejected", "reason": "invalid_task_result", "result": result})
            continue
        existing_index = next((index for index, item in enumerate(tasks) if item.get("task_id") == task_id), None)
        if task_id in processed and status in TERMINAL_STATUSES:
            reconciled.append({"status": "already_processed", "task_id": task_id})
            continue
        if existing_index is None:
            tasks.append(result)
        else:
            tasks[existing_index] = {**tasks[existing_index], **result}
        effective = current.setdefault("effective_decisions", {}).setdefault(task_key, {})
        effective["status"] = status
        if status in TERMINAL_STATUSES:
            reported_cost = result.get("actual_cost")
            if isinstance(reported_cost, bool) or not isinstance(reported_cost, (int, float)):
                if reported_cost is not None:
                    result["unparsed_actual_cost"] = reported_cost
                reported_cost = None
            reservation = current.get("budget_reservations", {}).get(task_key) or {}
            gpu = extract_gpu_accounting(result)
            result["actual_gpu_core_hours"] = gpu["actual_gpu_core_hours"]
            result["gpu_cost_source"] = gpu["source"]
            estimated_cost = result.get("estimated_cost")
            if isinstance(estimated_cost, bool) or not isinstance(estimated_cost, (int, float)):
                estimated_cost = None
            estimate_basis = result.get("estimated_cost_basis")
            if reported_cost is None and result.get("actual_mc_steps") is not None:
                requested = result.get("requested_max_mc_steps") or result.get("max_mc_steps")
                if isinstance(requested, (int, float)) and float(requested) > 0:
                    reserved_cost = float(reservation.get("reserved_cost", reservation.get("relative_cost", 0)) or 0)
                    estimated_cost = reserved_cost * min(1.0, max(0.0, float(result["actual_mc_steps"]) / float(requested)))
                    estimate_basis = "estimated_from_actual_steps_not_measured_cost"
                    result["actual_cost"] = None
                    result["estimated_cost"] = estimated_cost
                    result["estimated_cost_basis"] = estimate_basis
                    matched = next((row for row in tasks if row.get("task_id") == task_id), None)
                    if matched is not None:
                        matched.update({"actual_cost": None, "estimated_cost": estimated_cost,
                                        "estimated_cost_basis": estimate_basis,
                                        "actual_gpu_core_hours": gpu["actual_gpu_core_hours"],
                                        "gpu_cost_source": gpu["source"]})
            matched = next((row for row in tasks if row.get("task_id") == task_id), None)
            if matched is not None:
                matched.update({"actual_cost": reported_cost,
                                "estimated_cost": estimated_cost,
                                "estimated_cost_basis": estimate_basis,
                                "actual_gpu_core_hours": gpu["actual_gpu_core_hours"],
                                "gpu_cost_source": gpu["source"]})
            settlement = settle_budget(
                current,
                task_key=task_key,
                settlement_id=result.get("settlement_id") or task_id,
                task_status=status,
                actual_cost=reported_cost,
                estimated_cost=estimated_cost,
                cost_source=(result.get("actual_cost_basis") if reported_cost is not None else estimate_basis),
                failure_reason=result.get("failure_reason", result.get("error")),
            )
            current = settlement["state"]
            tasks = current.setdefault("tasks", tasks)
            processed = current.setdefault("processed_task_ids", processed)
            if task_id not in processed:
                processed.append(task_id)
            effective = current.setdefault("effective_decisions", {}).setdefault(task_key, {})
            effective.update({"status": status, "result_recorded": True})
            settled_record = current.setdefault("budget_reservations", {}).get(task_key) or {}
            settled_record.update({"actual_gpu_core_hours": gpu["actual_gpu_core_hours"],
                                   "gpu_cost_source": gpu["source"],
                                   "requested_max_mc_steps": result.get("requested_max_mc_steps"),
                                   "patience_steps": result.get("patience_steps"),
                                   "min_improvement": result.get("min_improvement"),
                                   "actual_mc_steps": result.get("actual_mc_steps"),
                                   "stop_reason": result.get("stop_reason")})
            reconciled.append({"status": "settled" if settlement["status"] in {"settled", "already_settled"} else settlement["status"], "task_id": task_id, "task_key": task_key})
        else:
            reservation = current.setdefault("budget_reservations", {}).get(task_key)
            if reservation is not None:
                reservation["status"] = "submitted" if status == "pending" else "running"
            reconciled.append({"status": status, "task_id": task_id, "task_key": task_key})
    current["pending_tasks"] = [item for item in tasks if item.get("status") in ACTIVE_STATUSES]
    return {"state": current, "reconciled": reconciled}

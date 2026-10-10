"""Create one budgeted retry task while preserving the failed attempt."""

from copy import deepcopy
import hashlib
from phase_agent.tools.budget.reserve_budget import reserve_budget


def restart_failed_task(state, task_id, *, budget_limits, config_version, max_retries=1):
    current = deepcopy(state)
    original = next(
        (row for row in current.get("tasks") or [] if row.get("task_id") == task_id), None
    )
    if original is None:
        return {"status": "rejected", "reason": "task_not_found", "state": current}
    if original.get("status") not in {"failed", "timeout"}:
        return {"status": "rejected", "reason": "task_not_retryable", "state": current}
    retry = int(original.get("retry_index", 0)) + 1
    if retry > int(max_retries):
        return {"status": "rejected", "reason": "retry_limit", "state": current}
    key = f"{original['task_key']}:retry:{retry}"
    if any(row.get("task_key") == key for row in current.get("tasks") or []):
        return {"status": "already_prepared", "state": current}
    cost = float(original.get("planned_relative_cost", original.get("budget", 0)) or 0)
    reservation = reserve_budget(
        current,
        task_key=key,
        stage=original["stage"],
        amount=cost,
        limits=budget_limits,
        config_version=config_version,
        model_version=original.get("model_version"),
    )
    if reservation["status"] != "reserved":
        return {
            "status": "budget_exhausted",
            "reason": reservation.get("reasons"),
            "state": current,
        }
    current = reservation["state"]
    task = {
        key_name: deepcopy(value)
        for key_name, value in original.items()
        if key_name
        not in {
            "job_id",
            "batch_id",
            "slurm_batch_id",
            "slurm_array_index",
            "input_path",
            "result_path",
            "outputs",
            "result",
            "error",
            "failure_reason",
        }
    }
    task.update(
        task_key=key,
        task_id="RETRY-" + hashlib.sha256(key.encode()).hexdigest()[:12],
        status="pending",
        retry_index=retry,
        retry_of=original["task_id"],
        config_version=config_version,
    )
    current.setdefault("tasks", []).append(task)
    current.setdefault("pending_tasks", []).append(task)
    return {"status": "prepared", "state": current, "task": task}

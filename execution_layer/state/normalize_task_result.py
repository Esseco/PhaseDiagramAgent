"""统一阶段执行结果，不用阶段编号推断状态。"""

from copy import deepcopy


def normalize_task_result(task: dict, result: dict) -> dict:
    status = result.get("status")
    allowed = {"pending", "running", "completed", "failed", "paused", "deferred", "unknown", "low_yield", "budget_exhausted", "not_configured", "cancelled", "timeout"}
    if status not in allowed:
        raise ValueError(f"非法运行状态：{status}")
    return {
        **deepcopy(task),
        **deepcopy(result),
        "task_id": task["task_id"],
        "object_id": task["object_id"],
        "structure_id": task["structure_id"],
        "stage": task["stage"],
        "status": status,
        "result": deepcopy(result.get("result", result.get("outputs") or {})),
        "outputs": deepcopy(result.get("outputs", result.get("result") or {})),
        "actual_cost": deepcopy(result.get("actual_cost")),
        "config_version": task.get("config_version"),
        "model_version": result.get("model_version", task.get("model_version")),
        "validity": deepcopy(result.get("validity") or {"status": "unknown" if status in {"unknown", "not_configured"} else "valid"}),
        "failure_reason": deepcopy(result.get("failure_reason", result.get("error"))),
    }

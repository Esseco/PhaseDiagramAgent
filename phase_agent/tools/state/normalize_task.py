"""统一计算任务输入，同时保留旧字段。"""

from copy import deepcopy


def normalize_task(task: dict) -> dict:
    object_id = task.get("object_id") or task.get("structure_id")
    if not object_id or not task.get("task_id") or not task.get("stage"):
        raise ValueError("任务必须包含 task_id、object_id/structure_id 和 stage")
    return {
        **deepcopy(task),
        "object_id": object_id,
        "structure_id": task.get("structure_id", object_id),
        "parameters": deepcopy(task.get("parameters") or {}),
        "budget": deepcopy(task.get("budget") or {}),
        "model_version": task.get("model_version", task.get("calculation_version")),
        "status": task.get("status", "pending"),
    }

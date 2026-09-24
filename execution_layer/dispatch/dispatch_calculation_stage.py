"""通过计算阶段注册表执行统一任务。"""

from execution_layer.state.normalize_task import normalize_task
from execution_layer.state.normalize_task_result import normalize_task_result


def execute_registered_stage(task: dict, context: dict, registry) -> dict:
    normalized = normalize_task(task)
    spec = registry.get(normalized["stage"])
    if spec.get("handler") is None:
        return normalize_task_result(normalized, {"status": "not_configured", "error": {"type": "NotConfigured", "message": f"阶段 {normalized['stage']} 没有执行器"}})
    try:
        return normalize_task_result(normalized, spec["handler"](normalized, context))
    except Exception as error:
        return normalize_task_result(normalized, {"status": "failed", "converged": False, "error": {"type": type(error).__name__, "message": str(error)}})

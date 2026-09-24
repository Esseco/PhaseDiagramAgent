"""仅在任务实际提交或 API 实际返回后记录用量。"""

from copy import deepcopy


def record_budget_usage(state: dict, actual: dict) -> dict:
    updated = deepcopy(state)
    usage = updated.setdefault("budget_usage", {})
    usage.setdefault("total_relative_cost", 0.0)
    usage.setdefault("stages", {})
    llm_usage = usage.setdefault("llm", {})
    for field in ("calls", "input_tokens", "output_tokens", "cost"):
        llm_usage.setdefault(field, 0)
    llm_usage.setdefault("calls_by_iteration", {})
    cost = float(actual.get("relative_cost", 0.0))
    usage["total_relative_cost"] += cost
    stage = actual.get("stage")
    if stage:
        item = usage["stages"].setdefault(stage, {"tasks": 0, "cost": 0.0})
        item["tasks"] += int(actual.get("tasks", 1))
        item["cost"] += cost
    llm = actual.get("llm_usage")
    if llm:
        for field in ("calls", "input_tokens", "output_tokens", "cost"):
            value = llm.get(field)
            if value is not None:
                usage["llm"][field] += value
        iteration = str(actual.get("iteration", 0))
        calls = usage["llm"]["calls_by_iteration"]
        calls[iteration] = calls.get(iteration, 0) + llm.get("calls", 0)
    return updated

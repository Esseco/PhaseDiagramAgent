"""根据 branch 搜索历史决定追加、暂停或结束。"""

from __future__ import annotations

from typing import Any


def check_branch_stop(
    branch_state: dict[str, Any], *, rules: dict[str, Any] | None = None
) -> dict[str, Any]:
    """未计算候选不作为失败；活动任务优先等待。"""
    config = {
        "minimum_completed_searches": 2,
        "max_no_improvement": 2,
        "minimum_energy_improvement": 1e-3,
        "max_failures": 2,
    }
    config.update(rules or {})
    tasks = branch_state.get("tasks", [])
    active = [item for item in tasks if item.get("status") in {"pending", "running"}]
    if active:
        return {
            "decision": "pause",
            "reason": "tasks_in_progress",
            "active_task_ids": [item.get("task_id") for item in active],
        }
    completed = [item for item in tasks if item.get("status") == "completed"]
    failed = [item for item in tasks if item.get("status") == "failed"]
    if (
        branch_state.get("budget_remaining") is not None
        and branch_state["budget_remaining"] <= 0
    ):
        return {"decision": "pause", "reason": "branch_budget_exhausted"}
    if len(failed) >= config["max_failures"]:
        return {"decision": "pause", "reason": "repeated_failures"}
    if len(completed) < config["minimum_completed_searches"]:
        return {
            "decision": "append_search",
            "reason": "insufficient_completed_searches",
        }
    improvements = [float(item.get("improvement", 0.0)) for item in completed]
    tail = improvements[-config["max_no_improvement"] :]
    if len(tail) == config["max_no_improvement"] and all(
        value < config["minimum_energy_improvement"] for value in tail
    ):
        return {"decision": "stop", "reason": "repeated_search_without_improvement"}
    return {"decision": "append_search", "reason": "potential_improvement_remains"}

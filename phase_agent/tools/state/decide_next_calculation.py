"""按可配置阶段图和显式运行状态决定下一动作。"""

from __future__ import annotations

from typing import Any


DEFAULT_STAGES = [
    {"name": "simple_check", "enabled": True, "max_retries": 1},
    {"name": "relax_and_feature", "enabled": True, "max_retries": 1},
    {"name": "deep_search", "enabled": True, "max_retries": 1},
    {"name": "dft_single_point", "enabled": True, "max_retries": 1},
    {"name": "dft_relax", "enabled": True, "max_retries": 1},
]


def decide_next_calculation(
    structure_record: dict[str, Any], *, rules: dict[str, Any] | None = None
) -> dict[str, Any]:
    config = {
        "stages": DEFAULT_STAGES,
        "active_statuses": ["pending", "running"],
        "skip_stages": [],
        "repeat_stages": [],
        "pause_on_failure": False,
        "allow_jump_to": None,
        "max_retries": 1,
        "resume_paused": False,
        "retry_not_configured": False,
    }
    config.update(rules or {})
    if (rules or {}).get("stage_order") is not None:
        config["stages"] = [
            {"name": name, "enabled": True, "max_retries": config["max_retries"]}
            for name in config["stage_order"]
        ]
    attempts = (structure_record.get("metadata") or {}).get("calculation_attempts", [])
    active = [item for item in attempts if item.get("status") in set(config["active_statuses"])]
    if active:
        task = active[-1]
        return {
            "action": "continue" if task["status"] == "running" else "run",
            "stage": task["stage"],
            "reason": f"existing_{task['status']}_task",
            "task_id": task.get("task_id"),
            "run_status": task["status"],
        }
    if attempts and attempts[-1].get("status") == "paused" and config["resume_paused"]:
        return {
            "action": "run",
            "stage": attempts[-1].get("stage"),
            "reason": "resume_paused_task",
            "task_id": attempts[-1].get("task_id"),
            "run_status": "ready",
        }
    if (
        attempts
        and attempts[-1].get("status") == "not_configured"
        and config["retry_not_configured"]
    ):
        return {
            "action": "retry",
            "stage": attempts[-1].get("stage"),
            "reason": "backend_now_configured",
            "task_id": attempts[-1].get("task_id"),
            "run_status": "ready",
        }
    if attempts and attempts[-1].get("status") in {"paused", "not_configured"}:
        status = attempts[-1]["status"]
        return {
            "action": "pause",
            "stage": attempts[-1].get("stage"),
            "reason": "task_is_paused" if status == "paused" else "backend_not_configured",
            "task_id": attempts[-1].get("task_id"),
            "run_status": status,
        }
    stages = [
        dict(item)
        for item in config["stages"]
        if item.get("enabled", True) and item["name"] not in set(config["skip_stages"])
    ]
    names = [item["name"] for item in stages]
    if config["allow_jump_to"] is not None:
        if config["allow_jump_to"] not in names:
            raise ValueError(f"未知跳级阶段：{config['allow_jump_to']}")
        stages = stages[names.index(config["allow_jump_to"]) :]
        names = [item["name"] for item in stages]
    history = structure_record.get("stage_history", {})
    stage_map = {item["name"]: item for item in stages}
    index = 0
    visited = set()
    while index < len(stages):
        spec = stages[index]
        name = spec["name"]
        if name in visited:
            raise ValueError(f"阶段跳转形成循环：{name}")
        visited.add(name)
        completed = any(item.get("converged") is not False for item in history.get(name, []))
        if completed and name not in set(config["repeat_stages"]):
            target = spec.get("on_success")
            index = names.index(target) if target in stage_map else index + 1
            continue
        failures = [
            item
            for item in attempts
            if item.get("stage") == name and item.get("status") == "failed"
        ]
        maximum = int(spec.get("max_retries", config["max_retries"]))
        if len(failures) > maximum:
            target = spec.get("on_failure")
            if target == "pause" or target is None:
                return {
                    "action": "pause",
                    "stage": name,
                    "reason": "retry_limit",
                    "run_status": "paused",
                }
            if target not in stage_map:
                raise ValueError(f"未知失败跳转阶段：{target}")
            index = names.index(target)
            continue
        if failures and config["pause_on_failure"]:
            return {
                "action": "pause",
                "stage": name,
                "reason": "failure_rule",
                "run_status": "paused",
            }
        return {
            "action": "run" if not failures and not completed else "retry",
            "stage": name,
            "reason": "repeat_rule"
            if completed
            else "failed_attempt"
            if failures
            else "next_incomplete_stage",
            "run_status": "ready",
        }
    return {
        "action": "complete",
        "stage": None,
        "reason": "all_required_stages_complete",
        "run_status": "completed",
    }

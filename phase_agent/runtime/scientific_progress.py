"""Serializable graph progress projected from durable project records."""

import json
from pathlib import Path


def scientific_progress(config_path):
    config = Path(config_path)
    settings = json.loads(config.read_text(encoding="utf-8"))
    if not settings.get("state_path"):
        return {}
    state_path = Path(settings["state_path"])
    if not state_path.is_absolute():
        state_path = config.parent / state_path
    if not state_path.is_file():
        return {}
    state = json.loads(state_path.read_text(encoding="utf-8"))
    progress = {
        "active_model_version": state.get("active_model_version"),
        "pending_action_count": len(state.get("pending_execution_policies") or {}),
        "directions": [],
    }
    for job in state.get("remote_finetune_jobs", {}).values():
        handoff = job.get("training_handoff") or {}
        proposal = handoff.get("direction_proposal")
        if proposal:
            progress["directions"].append(
                {
                    "proposal": proposal,
                    "status": handoff.get("direction_status"),
                    "waiting_at": handoff.get("stage"),
                    "activated": bool(job.get("activated")),
                }
            )
    pending = state.get("pending_execution_policies") or {}
    progress["execution_approvals"] = pending
    waiting = [
        row
        for row in progress["directions"]
        if row.get("status") == "awaiting_approval" and not row["activated"]
    ]
    progress["waiting_for"] = (
        "direction_approval"
        if waiting
        else "execution_approval"
        if pending
        else "project_input_or_results"
    )
    tasks = state.get("tasks") or []
    if isinstance(tasks, dict):
        tasks = list(tasks.values())
    active = any(row.get("status") in {"pending", "running", "submitted"} for row in tasks)
    fallback = (
        "已有任务待执行或回收；下一步核对任务结果"
        if active
        else "已有任务记录；下一步评估结果并提出方案"
        if tasks
        else "尚未开始搜索；下一步生成初始结构方案，批准后执行"
    )
    progress["summary"] = (
        "等待方向审批"
        if waiting
        else "等待执行方案审批"
        if pending
        else "模型已切换，等待后续处理"
        if any(row["activated"] for row in progress["directions"])
        else "已有方向判断，等待后续处理"
        if progress["directions"]
        else fallback
    )
    return progress

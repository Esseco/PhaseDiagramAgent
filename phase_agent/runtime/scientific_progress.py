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
    session = None
    if settings.get("config_session_path"):
        session_path = Path(settings["config_session_path"])
        if not session_path.is_absolute():
            session_path = config.parent / session_path
        if session_path.is_file():
            session = json.loads(session_path.read_text(encoding="utf-8"))
    return project_progress(state, state_path, session=session)


def project_progress(state, state_path, *, session=None):
    """One projection shared by Studio and the authenticated status endpoint."""
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
        if row.get("status") == "awaiting_approval"
        and row.get("waiting_at") != "execution_plan_ready"
        and not row["activated"]
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
    from phase_agent.runtime.review_requests import pending_reviews

    progress["review_requests"] = pending_reviews(state, session)
    progress["pending_review_count"] = len(progress["review_requests"])
    progress["plan_cards"] = [row["review_card"] for row in progress["review_requests"]]
    status, label, action = "ready", "可提出下一步方案", "回复继续，由 Agent 核对现有结果并提出方案"
    from phase_agent.graphs.execution_recovery_graph import execution_recovery_report

    report = execution_recovery_report(state_path, state)
    progress["recovery_report"] = report
    reconciliation = report.get("unsettled") or []
    if reconciliation:
        status, label, action = (
            "reconciliation",
            "需要核对中断执行",
            "核对已有文件、任务和账本，再决定恢复；不要重复批准",
        )
    elif progress["review_requests"]:
        status, label, action = (
            "human_review",
            "等待人工审阅",
            "查看具体方案后批准、修改或拒绝；继续不代表批准",
        )
    elif any(row.get("status") == "running" for row in tasks):
        status, label, action = (
            "task_running",
            "计算任务记录为运行中",
            "核对超算状态并回传结果；Studio 聊天状态不代表超算状态",
        )
    elif active:
        status, label, action = (
            "external_results",
            "等待任务执行或结果回传",
            "确认已提交任务，回传结果后回复继续；未回传结果不算完成",
        )
    progress["waiting_state"] = {"kind": status, "label": label, "next_action": action}
    progress["summary"] = label
    return progress

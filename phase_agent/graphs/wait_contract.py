"""Shared workflow wait boundaries; a resume signal never authorizes an action."""

WAIT_STATES = {
    "awaiting_approval": ("approval", "查看方案后批准、拒绝或提出修改"),
    "awaiting_direction_approval": ("approval", "查看科学方向与具体计划后审批"),
    "awaiting_manual_submission": ("results", "提交已准备输入并回传结果后重新核对"),
    "awaiting_remote_training": ("results", "回传训练结果后重新核对"),
    "tasks_in_progress": ("results", "等待或回传计算结果后重新核对"),
    "pending": ("results", "核对待执行任务，不重新派发"),
    "running": ("results", "等待任务结果，不重复执行"),
    "training_handoff": ("input", "查看训练交接要求，补齐结果或审批"),
    "confirmation_required": ("input", "核对具体确认事项后回复"),
    "execution_reconciliation_required": (
        "reconciliation",
        "核对原动作文件、任务与台账，不能直接重试",
    ),
    "approval_reconciliation_required": (
        "reconciliation",
        "核对已交付审批及执行结果，不能再次批准重放",
    ),
}


def wait_boundary(status):
    entry = WAIT_STATES.get(status)
    if entry is None and isinstance(status, str) and status.startswith("awaiting_"):
        entry = ("input", "按本轮提示补齐输入后重新核对")
    if entry is None:
        return None
    return {
        "kind": entry[0],
        "status": status,
        "resume_requirement": entry[1],
        "resume_authorizes_execution": False,
    }

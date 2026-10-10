"""Resume the exact pending non-activation scientific direction."""

from copy import deepcopy
from phase_agent.graphs.direction_review_graph import direction_checkpoint


def review_direction_command(message, state, state_path):
    if str(message).strip() not in {"同意", "拒绝"}:
        return None
    waits = [
        job
        for job in state.get("remote_finetune_jobs", {}).values()
        if (job.get("training_handoff") or {}).get("stage") == "awaiting_direction_approval"
    ]
    if not waits:
        return None
    if len(waits) != 1 or state.get("pending_execution_policies"):
        return {"state": state, "reason": "存在多个审批对象，请明确要批准的方案；未执行。"}
    updated = deepcopy(state)
    job = next(
        job
        for job in updated["remote_finetune_jobs"].values()
        if (job.get("training_handoff") or {}).get("stage") == "awaiting_direction_approval"
    )
    handoff = job["training_handoff"]
    from phase_agent.tools.local.recover_remote_training import inspect_training_results

    fresh = inspect_training_results(job)
    if (
        not fresh
        or fresh.get("fingerprint") != handoff["direction_proposal"]["training_fingerprint"]
    ):
        return {"state": state, "reason": "回传证据已变化，请继续重新判断；未批准旧方案。"}
    decision = "approve" if str(message).strip() == "同意" else "reject"
    result = direction_checkpoint(state_path, handoff["direction_proposal"], decision)
    handoff["direction_status"] = result["status"]
    handoff["stage"] = (
        "validation_prerequisites_required" if decision == "approve" else "direction_rejected"
    )
    handoff["reason"] = (
        "方向及计划已批准。回复‘继续’核对执行条件并准备输入执行审批；不会重新选择方向。"
        if decision == "approve"
        else "方向已拒绝，未执行计算。"
    )
    return {"state": updated, "reason": handoff["reason"]}

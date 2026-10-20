"""Read-only recovery facts; a historic controller error is not a task failure."""

from phase_agent.tools.state.task_waiting import awaiting_task_result
from phase_agent.tools.relax_mc_batch.handoff import result_archive


def result_recovery_facts(state):
    tasks = state.get("tasks") or []
    rows = []
    for identity, batch in (state.get("relax_mc_batches") or {}).items():
        if batch.get("status") in {"cancelled", "superseded"} or batch.get("model_version") not in (
            None,
            state.get("active_model_version"),
        ):
            continue
        members = [t for t in tasks if t.get("relax_mc_bundle_id") == identity]
        pending = [t["task_id"] for t in members if awaiting_task_result(t)]
        summary = batch.get("result_summary") or {}
        recovery = (
            summary.get("recovery") or (state.get("result_recoveries") or {}).get(identity) or {}
        )
        try:
            returned = result_archive(batch).is_file()
        except (KeyError, ValueError, OSError):
            returned = False
        if batch.get("return_error"):
            kind, label = "blocked", "本地回收校验受阻，保留原包；未重投计算"
        elif (
            batch.get("import_checksum")
            and not pending
            and recovery.get("next_phase_not_submitted")
        ):
            kind, label = "next_step_review", "当前阶段结果已回收，后续阶段尚未提交，待核对方案"
        elif batch.get("import_checksum") and not pending and summary.get("failed_task_ids"):
            kind, label = "needs_review", "已核验入账，存在失败任务待审阅；未重投计算"
        elif batch.get("import_checksum") and not pending:
            kind, label = "completed", "本批已核验入账；计算完成不代表科学收敛"
        elif batch.get("import_checksum"):
            kind, label = "awaiting_verification", "部分结果已入账，剩余结果/作业状态待核验"
        elif returned:
            kind, label = "awaiting_verification", "回传包已到，等待本地校验入账"
        else:
            kind, label = "external_results", "等待回传；尚未核实超算当前状态"
        rows.append(
            {
                "bundle_id": identity,
                "kind": kind,
                "label": label,
                "pending_task_ids": pending,
                "pending_count": len(pending),
                "verified_failed_task_ids": list(summary.get("failed_task_ids") or []),
                "registered_task_count": len(members),
                "verified_archive_checksum": batch.get("import_checksum"),
                "current_error": batch.get("return_error"),
                "historical_controller_diagnostic": summary.get("reason"),
                "recovered_task_count": len(recovery.get("recovered_task_ids") or []),
                "job_evidence": recovery.get("job_evidence"),
                "submitted": False,
            }
        )
    return rows

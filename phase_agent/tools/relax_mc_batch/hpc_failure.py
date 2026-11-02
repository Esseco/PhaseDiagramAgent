"""Evidence-based startup failure facts shared by the Agent, MCP and resume runner."""

import shlex


def classify_scheduler_failure(row):
    if row.get("query_error"):
        return "unknown"
    if type(row.get("elapsed_seconds")) in (int, float) and row["elapsed_seconds"] == 0:
        if row.get("state") == "FAILED" and row.get("exit_code") == "0:53":
            return "hpc_launch_io"
        if row.get("state") in {"BOOT_FAIL", "NODE_FAIL"}:
            return "hpc_startup_node_failure"
    return "unclassified"


def startup_recovery_options(state):
    options = []
    model = state.get("active_model_version")
    for bundle_id, batch in (state.get("relax_mc_batches") or {}).items():
        if model and batch.get("model_version") != model:
            continue
        summary = batch.get("result_summary") or {}
        if summary.get("status") not in {"paused", "failed", "partial"}:
            continue
        diagnostic = batch.get("remote_diagnostics") or summary.get("diagnostics") or {}
        jobs = diagnostic.get("jobs") or []
        affected = []
        for job in jobs:
            category = classify_scheduler_failure(job)
            if category.startswith("hpc_") or (
                job.get("state") == "FAILED" and job.get("exit_code") == "0:53"
            ):
                affected.append({**job, "failure_category": category})
        if not affected:
            continue
        ids = {str(j["job_id"]) for j in affected}
        requested = {
            t
            for group in diagnostic.get("issue_groups", [])
            if str(group.get("job_id")) in ids
            for t in group.get("task_ids", [])
        }
        tasks = [t for t in state.get("tasks", []) if t.get("relax_mc_bundle_id") == bundle_id]
        completed = {t["task_id"] for t in tasks if t.get("status") == "completed"}
        requested -= completed
        roots = {
            str(t.get("calculation_result_path")).split("/jobs/", 1)[0]
            for t in tasks
            if str(t.get("calculation_result_path", "")).startswith("/")
            and "/jobs/" in str(t.get("calculation_result_path"))
        }
        remote = next(iter(roots)) if len(roots) == 1 else None
        options.append(
            {
                "bundle_id": bundle_id,
                "kind": "hpc_startup_failure",
                "job_evidence": affected,
                "candidate_task_ids": sorted(requested),
                "candidate_task_count": len(requested),
                "preserve_completed_task_count": len(completed),
                "remote_work_root": remote,
                "remote_preview_command": "python3 run_batch.py resume --root "
                + shlex.quote(remote)
                if remote
                else None,
                "operation": "prepare_relax_mc_resume",
                "requires_remote_preflight": True,
                "instruction": "通过原resume入口核验调度器、输入和结果后，只补无结果的启动失败项；不微调、不重建整包、不重跑已完成项。状态/回执不明必须阻断。提交仍遵守既有授权。",
            }
        )
    return options


def prepare_resume_handoff(*, action, context):
    bundle_id = (action.get("parameters") or {}).get("bundle_id")
    facts = next(
        (
            r
            for r in startup_recovery_options(context.get("event_state") or {})
            if r["bundle_id"] == bundle_id
        ),
        None,
    )
    if not facts or not facts["candidate_task_ids"] or not facts["remote_work_root"]:
        raise ValueError("HPC_startup_recovery_evidence_required")
    return {
        "status": "prepared",
        "submitted": False,
        "recomputed": False,
        "recovery_plan": facts,
        "next_action": "在已登记远端work执行原resume预检；以预检plan_id按已授权范围提交，不能跳过远端校验。",
    }

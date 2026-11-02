"""Preview-bound recovery of whole batches that failed before producing any result."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from phase_agent.tools.relax_mc_batch.files import read, rebase
from phase_agent.tools.relax_mc_batch.run import SlurmIO, checkpoint, validate_frozen_inputs
from phase_agent.tools.remote.integrity import file_checksum, verify_result
from phase_agent.tools.remote.transfer_bundle import _archive_lock
from phase_agent.persistence.ledger.phase_data_manager import PhaseDataManager


FAILED = {"FAILED", "NODE_FAIL", "BOOT_FAIL", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"}


def _plan(root, manifest, state, io, exclude_nodes):
    meta = state.get("relax_mc_remote") or {}
    phase = meta.get("phase")
    if meta.get("status") not in {"paused", "partial", "failed"} or phase not in {1, 2, 3}:
        raise ValueError("resume_requires_paused_GPU_phase")
    control = (meta.get("phase_controllers") or {}).get(str(phase)) or {}
    batches = [
        b for b in state.get("slurm_batches", []) if b["batch_id"] in control.get("batch_ids", [])
    ]
    if not batches or {b["batch_id"] for b in batches} != set(control["batch_ids"]):
        raise ValueError("resume_phase_batch_scope_invalid")
    tasks = {t["task_id"]: t for t in state.get("tasks", [])}
    ids = [str(b["job_id"]) for b in batches]
    if io.active(ids):
        raise ValueError("resume_jobs_still_active")
    evidence = io.evidence(ids)
    retry, preserved = [], []
    for batch in batches:
        missing, completed = [], []
        for tid in batch["task_ids"]:
            task = tasks[tid]
            if task.get("relax_mc_bundle_id") != manifest["bundle_id"]:
                raise ValueError("resume_task_bundle_mismatch")
            path = Path(task["result_path"])
            result = path if path.name == "result.json" else path.parent / "result.json"
            marker = result.with_name("task.finished.json")
            worker = Path(task["input_path"]).parent
            if task.get("status") == "completed":
                checked = verify_result(result, marker, task)
                if not checked["valid"] or checked["result"].get("status") != "completed":
                    raise ValueError("resume_completed_result_invalid:" + tid)
                completed.append({"task_id": tid, "result_sha256": file_checksum(result)})
                continue
            # A partial or uncollected worker result must be inspected, never overwritten.
            if any(
                p.exists()
                for p in [result, marker, worker / "result.json", worker / "task.finished.json"]
            ):
                raise ValueError("resume_existing_worker_result_requires_review:" + tid)
            payload = read(task["input_path"])
            for key in ("task_id", "task_key", "task_checksum", "model_version", "config_version"):
                if payload.get(key) != task.get(key):
                    raise ValueError("resume_task_input_binding_mismatch:" + tid + ":" + key)
            missing.append(tid)
        preserved.extend(completed)
        if not missing:
            continue
        if completed:
            raise ValueError("resume_mixed_batch_requires_subset_plan")
        job = str(batch["job_id"])
        from phase_agent.tools.relax_mc_batch.hpc_failure import classify_scheduler_failure

        if not classify_scheduler_failure(evidence.get(job, {})).startswith("hpc_"):
            raise ValueError("resume_noncompute_startup_failure_not_confirmed:" + job)
        if evidence.get(job, {}).get("state") not in FAILED:
            raise ValueError("resume_job_failure_not_confirmed:" + job)
        script = Path(batch["script_path"]).resolve()
        script.relative_to(root)
        retry.append(
            {
                "batch_id": batch["batch_id"],
                "old_job_id": job,
                "task_ids": sorted(missing),
                "script": str(script),
                "script_sha256": file_checksum(script),
                "old_outcome": evidence[job],
            }
        )
    if not retry:
        raise ValueError("resume_no_startup_failed_batches")
    plan = {
        "bundle_id": manifest["bundle_id"],
        "phase": phase,
        "retry_batches": retry,
        "retry_task_count": sum(len(b["task_ids"]) for b in retry),
        "preserved_phase_tasks": preserved,
        "exclude_nodes": list(exclude_nodes),
    }
    plan["plan_id"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    return plan


def resume_batch(root, *, submit=False, expected_plan=None, exclude_nodes=(), io=None):
    """Never rerun a completed batch or retry an uncertain submission receipt."""
    root = Path(root).resolve()
    exclude_nodes = tuple(sorted(set(exclude_nodes)))
    if any(not re.fullmatch(r"[A-Za-z0-9_.-]+", node) for node in exclude_nodes):
        raise ValueError("invalid_excluded_node")
    io = io or SlurmIO()
    with _archive_lock(root):
        validate_frozen_inputs(root)
        manifest = rebase(read(root / "batch.json"), root)
        state = rebase(read(root / "workflow_state.json"), root)
        meta = state["relax_mc_remote"]
        previous = meta.get("resume_attempt") or {}
        if expected_plan and previous.get("plan_id") == expected_plan:
            if previous.get("status") != "queued":
                raise ValueError("resume_submission_uncertain; inspect recorded job IDs; no retry")
            return {**previous, "status": "already_queued", "submitted": False}
        plan = _plan(root, manifest, state, io, exclude_nodes)
        if not submit:
            return {"status": "preview", "submitted": False, **plan}
        if expected_plan != plan["plan_id"]:
            raise ValueError("resume_plan_changed; preview_required")
        data = rebase(read(root / "manager_checkpoint.json"), root)
        manager = PhaseDataManager(data["boundary"], data.get("system_config"))
        manager.data = data
        meta["resume_attempt"] = {
            "plan_id": expected_plan,
            "status": "submitting",
            "gpu_job_ids": [],
            "retry_task_count": plan["retry_task_count"],
        }
        attempt = meta["resume_attempt"]
        control = meta["phase_controllers"][str(meta["phase"])]
        control.setdefault("attempt_history", []).append(
            {k: deepcopy(v) for k, v in control.items() if k != "attempt_history"}
        )
        control.pop("job_id", None)
        control.pop("submission_state", None)
        meta["status"] = "resuming"
        checkpoint(root, state, manager)
        by_id = {b["batch_id"]: b for b in state["slurm_batches"]}
        try:
            for row in plan["retry_batches"]:
                batch = by_id[row["batch_id"]]
                batch.setdefault("attempt_history", []).append(
                    {
                        "job_id": row["old_job_id"],
                        "outcome": row["old_outcome"],
                        "reason": "approved_startup_failure_retry",
                    }
                )
                batch["submission_state"] = "submitting"
                batch.pop("job_id", None)
                checkpoint(root, state, manager)
                kwargs = {"exclude_nodes": exclude_nodes} if exclude_nodes else {}
                job = io.submit(Path(row["script"]), **kwargs)
                batch.update(job_id=job, submission_state="submitted")
                attempt["gpu_job_ids"].append(job)
                checkpoint(root, state, manager)
            meta["status"] = "running"
            meta.pop("failure_diagnostics", None)
            meta.pop("scheduler_query_error", None)
            checkpoint(root, state, manager)
            from phase_agent.tools.relax_mc_batch.dependencies import submit_analysis

            batches = [by_id[b] for b in control["batch_ids"]]
            outcome = submit_analysis(root, state, manager, manifest, io, batches)
            attempt.update(status="queued", analysis_job_id=outcome["analysis_job_id"])
            checkpoint(root, state, manager)
            return {"status": "queued", "submitted": True, **attempt}
        except Exception:
            attempt["status"] = "submission_uncertain"
            checkpoint(root, state, manager)
            raise

"""Short CPU phase controllers; Slurm owns waiting for whole GPU phases."""

from copy import deepcopy
from pathlib import Path
import shlex
import traceback

from phase_agent.persistence.ledger.phase_data_manager import PhaseDataManager
from phase_agent.tools.remote.transfer_bundle import _archive_lock
from phase_agent.tools.relax_mc_batch.files import (
    read,
    write,
    rebase,
    script,
    coordinator_profile,
    python_environment,
)
from phase_agent.tools.relax_mc_batch.analysis import feedback, next_mc
from phase_agent.tools.relax_mc_batch.run import (
    checkpoint,
    prepare_wave,
    publish,
    SchedulerQueryError,
    completion_outcome,
)

from phase_agent.tools.relax_mc_batch.failure_diagnostics import unverified_task_ids


def finish(root, state, manager, manifest, status, reason=None):
    status, reason = completion_outcome(state, status, reason)
    state["relax_mc_remote"]["status"] = status
    checkpoint(root, state, manager)
    archive = publish(root, state, manifest, status, reason)
    from phase_agent.tools.relax_mc_batch.upload_driver import publish_output

    publish_output(root, {"status": status, "reason": reason})
    return {"status": status, "reason": reason, "archive": archive, "submitted": True}


def submit_phase(root, state, manager, manifest, io):
    state, _ = prepare_wave(root, state, manager, manifest)
    meta = state["relax_mc_remote"]
    phase = meta["phase"]
    control = meta.setdefault("phase_controllers", {}).setdefault(str(phase), {})
    batches = [b for b in state.get("slurm_batches", []) if not b.get("relax_mc_wave_done")]
    if not batches:
        raise RuntimeError("empty_GPU_phase; no dependent job submitted")
    control["batch_ids"] = [b["batch_id"] for b in batches]
    checkpoint(root, state, manager)
    for batch in batches:
        if batch.get("submission_state") == "submitting" and not batch.get("job_id"):
            raise RuntimeError("submission_receipt_uncertain; inspect Slurm before retry")
        if batch.get("job_id"):
            continue
        batch["submission_state"] = "submitting"
        checkpoint(root, state, manager)
        batch["job_id"] = io.submit(batch["script_path"])
        batch["submission_state"] = "submitted"
        checkpoint(root, state, manager)
    if control.get("submission_state") == "submitting" and not control.get("job_id"):
        raise RuntimeError("analysis_receipt_uncertain; inspect Slurm before retry")
    if not control.get("job_id"):
        path = root / f"analyze_phase_{phase}.sh"
        command = (
            python_environment(root)
            + f"cd -- {shlex.quote(str(root))}\n"
            + f"python3 run_batch.py run --root {shlex.quote(str(root))} --submit --phase {phase}"
        )
        path.write_text(
            script(
                coordinator_profile(manifest["stage_profiles"]),
                command,
                name=manifest["bundle_id"] + f"-analysis-{phase}",
            ),
            encoding="utf-8",
            newline="\n",
        )
        control["submission_state"] = "submitting"
        checkpoint(root, state, manager)
        control["job_id"] = io.submit(
            path, dependency="afterany:" + ":".join(b["job_id"] for b in batches)
        )
        control["submission_state"] = "submitted"
        checkpoint(root, state, manager)
    return {
        "status": "queued",
        "phase": phase,
        "submitted": True,
        "gpu_job_ids": [b["job_id"] for b in batches],
        "analysis_job_id": control["job_id"],
    }


def collect_phase(root, state, manager, manifest, io):
    meta = state["relax_mc_remote"]
    control = meta["phase_controllers"][str(meta["phase"])]
    batches = [b for b in state["slurm_batches"] if b["batch_id"] in control["batch_ids"]]
    expected = {tid for batch in batches for tid in batch["task_ids"]}
    if (
        not expected
        or {b["batch_id"] for b in batches} != set(control["batch_ids"])
        or any(not b["task_ids"] for b in batches)
        or not expected <= {t["task_id"] for t in state.get("tasks") or []}
    ):
        raise ValueError(
            "phase_task_scope_invalid; inspect original job/task mapping; no submission"
        )
    ids = [b["job_id"] for b in batches]
    scheduler_unknown = False
    try:
        if io.active(ids):
            raise RuntimeError("GPU_phase_still_active; analysis must wait for afterany")
    except SchedulerQueryError as error:
        scheduler_unknown = True
        meta["scheduler_query_error"] = {"job_ids": ids, "diagnostics": error.diagnostics}
    else:
        meta.pop("scheduler_query_error", None)
    # Reuse receipt/checksum validation; no scientific task writes shared workflow state.
    from phase_agent.tools.remote.manual_upload_runner import ManualUploadBatchRunner

    runner = ManualUploadBatchRunner(
        root / "jobs",
        worker_command=manifest["config"]["supercomputer"]["worker"]["command"],
    )
    results = runner.collect_results(state)
    found = {r["task_id"]: r for r in results if r["task_id"] in expected}
    missing = expected - found.keys()
    from phase_agent.tools.relax_mc_batch.failure_diagnostics import record_remote_diagnostics

    results = list(found.values())
    if results:
        state = feedback(
            state,
            results,
            manager,
            manifest["phase_references"],
            root,
            manifest["model_version"],
        )
    missing |= unverified_task_ids(state, expected)
    record_remote_diagnostics(
        root, state, results, missing, getattr(runner, "last_collection_report", {}), io
    )
    if scheduler_unknown:
        state["relax_mc_remote"]["scheduler_query_error"] = {
            **meta["scheduler_query_error"],
            "unverified_task_ids": sorted(missing),
        }
        return state, results
    if missing:
        return state, results
    for batch in state["slurm_batches"]:
        if batch["batch_id"] in control["batch_ids"]:
            batch["relax_mc_wave_done"] = True
    return state, results


def advance(root, manifest, io, *, phase=None):
    """Entry submits Relax; dependent analysis consumes exactly one completed phase."""
    root = Path(root)
    with _archive_lock(root):
        path = root / "workflow_state.json"
        state = rebase(read(path), root) if path.exists() else deepcopy(manifest["initial_state"])
        data_path = root / "manager_checkpoint.json"
        data = rebase(read(data_path), root) if data_path.exists() else manifest["manager"]
        manager = PhaseDataManager(data["boundary"], data.get("system_config"))
        manager.data = data
        meta = state.setdefault("relax_mc_remote", {"phase": 0, "completed_mc_waves": []})
        if meta.get("status") in {"completed", "paused", "partial", "failed"}:
            return {"status": meta["status"], "submitted": False}
        if phase is not None and phase != meta["phase"]:
            return {"status": "stale_phase", "submitted": False}
        # A repeated entry may finish a partially submitted phase, but never collect it.
        meta["status"] = "running"
        checkpoint(root, state, manager)
        try:
            if phase is not None:
                state, results = collect_phase(root, state, manager, manifest, io)
                meta = state["relax_mc_remote"]
                if meta.get("scheduler_query_error"):
                    write(root / "scheduler_diagnostics.json", meta["scheduler_query_error"])
                    return finish(
                        root,
                        state,
                        manager,
                        manifest,
                        "partial",
                        "scheduler_state_unknown; verified results retained; no next phase submitted; inspect recorded job IDs before resuming",
                    )
                if (meta.get("failure_diagnostics") or {}).get("unverified_task_ids"):
                    return finish(
                        root,
                        state,
                        manager,
                        manifest,
                        "paused",
                        "remote_results_unverified; see return_diagnostics.json; no next phase submitted",
                    )
                if any(r.get("status") != "completed" for r in results):
                    return finish(
                        root,
                        state,
                        manager,
                        manifest,
                        "paused",
                        "Phase failures/missing results require inspection; no next stage submitted",
                    )
                if phase:
                    meta["completed_mc_waves"].append(phase - 1)
                meta["phase"] = phase + 1
                checkpoint(root, state, manager)
                if meta["phase"] >= 4:
                    return finish(root, state, manager, manifest, "completed")
                plan = next_mc(state, manager, manifest, root, meta["phase"] - 1)
                state = plan["state"]
                if not plan["actions"]:
                    status = (
                        "paused"
                        if plan["status"]
                        in {
                            "phase_diagram_unavailable",
                            "continuation_pool_below_minimum",
                            "tasks_in_progress",
                        }
                        else "completed"
                    )
                    return finish(
                        root,
                        state,
                        manager,
                        manifest,
                        status,
                        plan.get("reason", plan["status"]),
                    )
            return submit_phase(root, state, manager, manifest, io)
        except Exception as error:
            reason = f"{type(error).__name__}: {error}"
            write(
                root / "failure.json",
                {"reason": reason, "traceback": traceback.format_exc()},
            )
            # Preserve submission intents if a receipt is unknown. Never resubmit blindly.
            state = rebase(read(path), root)
            return finish(root, state, manager, manifest, "failed", reason)

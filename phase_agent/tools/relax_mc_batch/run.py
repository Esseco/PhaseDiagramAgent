"""Fixed-rule remote composition with existing task materialization and scientific modules."""

from copy import deepcopy
from pathlib import Path
import shlex
import subprocess
import time
import traceback

from phase_agent.tools.relax_mc_batch.files import (
    read,
    write,
    rebase,
    script,
    gpu_profile,
    python_environment,
)
from phase_agent.tools.relax_mc_batch.analysis import feedback, next_mc, candidate_pool
from phase_agent.persistence.ledger.phase_data_manager import PhaseDataManager
from phase_agent.tools.remote.manual_upload_runner import ManualUploadBatchRunner
from phase_agent.science.mlip.slurm_executor import create_mlip_task_preparer
from phase_agent.tools.remote.submission_dispatcher import preflight_models
from phase_agent.tools.remote.transfer_bundle import create_bundle, _archive_lock
from phase_agent.tools.remote.integrity import file_checksum


class SlurmIO:
    """Thin I/O adapter; existing runner owns tasks/batches, no second scheduler."""

    def submit(self, path, *, dependency=None):
        command = ["sbatch", "--parsable"]
        if dependency:
            command.append("--dependency=" + dependency)
        command.append(str(path))
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(path).parent,
        )
        job = result.stdout.strip().split(";")[0]
        if not job.isdigit():
            raise ValueError("invalid_sbatch_receipt; inspect scheduler before retry")
        return job

    def active(self, ids):
        if not ids:
            return set()
        result = subprocess.run(
            ["squeue", "--noheader", "--jobs", ",".join(ids), "--format=%A"],
            check=True,
            capture_output=True,
            text=True,
        )
        return set(result.stdout.split())


def checkpoint(root, state, manager):
    write(Path(root) / "workflow_state.json", rebase(state, root, portable=True))
    write(
        Path(root) / "manager_checkpoint.json",
        rebase(manager.data, root, portable=True),
    )


def prepare_wave(root, state, manager, manifest):
    config = manifest["config"]
    runner = ManualUploadBatchRunner(
        Path(root) / "jobs",
        worker_command=config["supercomputer"]["worker"]["command"],
        stage_batch_sizes=config["supercomputer"]["batch_sizes"],
        stage_profiles=manifest["stage_profiles"],
        task_preparer=create_mlip_task_preparer(manager, manifest["phase_references"], config),
    )
    while True:
        prepared = runner.prepare(state)
        state = prepared["state"]
        if prepared["status"] == "no_tasks":
            break
        batch = state["slurm_batches"][-1]
        batch_dir = Path(batch["upload_directory"])
        command = (
            python_environment(root)
            + f"cd -- {shlex.quote(str(batch_dir))}\npython3 run_mlip_batch.py"
        )
        Path(batch["script_path"]).write_text(
            script(
                gpu_profile(
                    manifest["stage_profiles"],
                    next(
                        t["stage"] for t in state["tasks"] if t["task_id"] == batch["task_ids"][0]
                    ),
                ),
                command,
                name=manifest["bundle_id"] + "-" + batch["batch_id"],
            ),
            encoding="utf-8",
            newline="\n",
        )
        checkpoint(root, state, manager)
    preflight_models(
        Path(root),
        [
            Path(row["script_path"]).relative_to(root).as_posix()
            for row in state.get("slurm_batches") or []
        ],
    )
    return state, runner


def wave(
    root,
    state,
    manager,
    manifest,
    io,
    *,
    batch_executor=None,
    poll_seconds=30,
    max_wait_seconds=604800,
):
    state, runner = prepare_wave(root, state, manager, manifest)
    if batch_executor is not None:
        results = []
        for batch in state.get("slurm_batches") or []:
            if batch.get("relax_mc_wave_done"):
                continue
            results.extend(batch_executor(batch, state, manager))
            batch["relax_mc_wave_done"] = True
        return state, results
    deadline = time.monotonic() + max_wait_seconds
    pending = [row for row in state.get("slurm_batches") or [] if not row.get("relax_mc_wave_done")]
    concurrency = manifest["workload"]["max_concurrent_jobs"]
    if manifest["workload"].get("submission_policy") == "whole_wave":
        # Missing marker preserves legacy packages' original submission limit.
        concurrency = len(pending)
    collected = {}
    while pending:
        results = runner.collect_results(state)
        collected.update({r["task_id"]: r for r in results})
        for batch in pending:
            if set(batch["task_ids"]) <= set(collected):
                batch["relax_mc_wave_done"] = True
        pending = [row for row in pending if not row.get("relax_mc_wave_done")]
        active_ids = io.active([row["job_id"] for row in pending if row.get("job_id")])
        for batch in pending:
            if batch.get("submission_state") == "submitting" and not batch.get("job_id"):
                raise RuntimeError(
                    "submission_receipt_uncertain; inspect Slurm before retry, no duplicate submission"
                )
            if batch.get("job_id") and batch["job_id"] not in active_ids:
                collected.update({r["task_id"]: r for r in runner.collect_results(state)})
                absent = set(batch["task_ids"]) - set(collected)
                if absent:
                    batch["missing_receipt_polls"] = batch.get("missing_receipt_polls", 0) + 1
                    if batch["missing_receipt_polls"] < 3:
                        continue
                # All queue jobs have ended: missing verified results is a failure, not a retry.
                for tid in set(batch["task_ids"]) - set(collected):
                    task = next(t for t in state["tasks"] if t["task_id"] == tid)
                    collected[tid] = {
                        **task,
                        "status": "failed",
                        "error": "scheduler_ended_without_verified_result",
                    }
                batch["relax_mc_wave_done"] = True
            elif not batch.get("job_id") and len(active_ids) < concurrency:
                batch["submission_state"] = "submitting"
                checkpoint(root, state, manager)
                batch["job_id"] = io.submit(batch["script_path"])
                batch["submission_state"] = "submitted"
                active_ids.add(batch["job_id"])
                checkpoint(root, state, manager)
        pending = [row for row in pending if not row.get("relax_mc_wave_done")]
        if pending:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    "remote_wait_timeout; existing jobs retained, resume after inspection"
                )
            time.sleep(poll_seconds)
    return state, list(collected.values())


def publish(root, state, manifest, status, reason=None):
    root = Path(root)
    write(
        root / "result_summary.json",
        {
            "bundle_id": manifest["bundle_id"],
            "plan_checksum": manifest["plan_checksum"],
            "model_version": manifest["model_version"],
            "config_version": manifest["config_version"],
            "status": status,
            "reason": reason,
            "task_count": len(state.get("tasks") or []),
            "gpu_job_count": len(state.get("slurm_batches") or []),
            "completed_mc_waves": (state.get("relax_mc_remote") or {}).get(
                "completed_mc_waves", []
            ),
            "failed_task_ids": [
                r["task_id"] for r in state.get("tasks") or [] if r.get("status") == "failed"
            ],
        },
    )
    write(root / "candidate_pool.json", rebase(candidate_pool(state), root, portable=True))
    logs = []
    for path in (root / "jobs").rglob("*"):
        if not path.is_file() or not (
            path.name in {"out", "err"} or path.suffix in {".out", ".err", ".log"}
        ):
            continue
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 65536))
            tail = stream.read().decode("utf-8", errors="replace")
        logs.append(
            {
                "path": path.relative_to(root).as_posix(),
                "tail": tail,
                "truncated": path.stat().st_size > 65536,
            }
        )
    write(root / "remote_logs.json", logs)
    files = [
        root / "workflow_state.json",
        root / "result_summary.json",
        root / "candidate_pool.json",
        root / "remote_logs.json",
    ]
    if (root / "failure.json").is_file():
        files.append(root / "failure.json")
    files.extend(
        p
        for name in ("analysis", "jobs", "inputs", "references")
        for p in (root / name).rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".json", ".gz", ".vasp", ".bin", ".csv"}
        and "runtime" not in p.parts
        and p.name not in {"task.json", "manifest.json", "batch.snapshot.json"}
    )
    return create_bundle(root, files, root / "results.zip", kind="results")


def run_batch(
    root,
    *,
    submit=False,
    dry_run=False,
    io=None,
    batch_executor=None,
    poll_seconds=30,
    phase=None,
):
    root = Path(root).resolve()
    manifest = rebase(read(root / "batch.json"), root)
    if dry_run or not submit:
        return {
            "status": "planned_only",
            "submitted": False,
            "workload": manifest["workload"],
        }
    for row in read(root / "transfer_manifest.json")["files"]:
        path = (root / row["path"]).resolve()
        if (
            not path.is_relative_to(root)
            or not path.is_file()
            or file_checksum(path) != row["sha256"]
        ):
            raise ValueError("input_package_integrity_failed; no jobs submitted")
    if (
        manifest["workload"].get("submission_policy") == "phase_dependencies"
        and batch_executor is None
    ):
        from phase_agent.tools.relax_mc_batch.dependencies import advance

        return advance(root, manifest, io or SlurmIO(), phase=phase)
    with _archive_lock(root):
        state_file = root / "workflow_state.json"
        state = (
            rebase(read(state_file), root)
            if state_file.is_file()
            else deepcopy(manifest["initial_state"])
        )
        data = (
            rebase(read(root / "manager_checkpoint.json"), root)
            if (root / "manager_checkpoint.json").exists()
            else manifest["manager"]
        )
        manager = PhaseDataManager(data["boundary"], data.get("system_config"))
        manager.data = data
        meta = state.setdefault("relax_mc_remote", {"phase": 0, "completed_mc_waves": []})
        if meta.get("status") in {"completed", "paused", "partial"}:
            return {
                "status": meta["status"],
                "archive": str(root / "results.zip"),
                "submitted": False,
            }
        meta["status"] = "running"
        status, reason = "completed", None
        try:
            for phase in range(int(meta["phase"]), 4):
                if phase:
                    if not any(t.get("status") == "pending" for t in state["tasks"]):
                        plan = next_mc(state, manager, manifest, root, phase - 1)
                        state = plan["state"]
                        if not plan["actions"]:
                            status = (
                                "paused"
                                if plan["status"]
                                in {
                                    "phase_diagram_unavailable",
                                    "continuation_pool_below_minimum",
                                }
                                else "completed"
                            )
                            reason = plan.get("reason", plan["status"])
                            break
                state, results = wave(
                    root,
                    state,
                    manager,
                    manifest,
                    io or SlurmIO(),
                    batch_executor=batch_executor,
                    poll_seconds=poll_seconds,
                )
                state = feedback(
                    state,
                    results,
                    manager,
                    manifest["phase_references"],
                    root,
                    manifest["model_version"],
                )
                meta = state["relax_mc_remote"]
                if phase:
                    meta["completed_mc_waves"].append(phase - 1)
                meta["phase"] = phase + 1
                checkpoint(root, state, manager)
                if phase and any(row.get("status") != "completed" for row in results):
                    status, reason = (
                        "paused",
                        "MC failures require inspection; no automatic resubmission",
                    )
                    break
        except Exception as error:
            status, reason = "failed", f"{type(error).__name__}: {error}"
            write(
                root / "failure.json",
                {"reason": reason, "traceback": traceback.format_exc()},
            )
            # wave checkpoints submission intents/receipts before returning. Never
            # overwrite them with the caller's older copy after an exception.
            if state_file.is_file():
                state = rebase(read(state_file), root)
        if status == "completed" and any(
            t.get("status") == "failed" for t in state.get("tasks") or []
        ):
            status = "partial"
        state["relax_mc_remote"]["status"] = status
        checkpoint(root, state, manager)
        archive = publish(root, state, manifest, status, reason)
        return {
            "status": status,
            "reason": reason,
            "archive": archive,
            "submitted": True,
        }

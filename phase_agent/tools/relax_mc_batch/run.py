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


class SchedulerQueryError(RuntimeError):
    """A query failure is not evidence that a GPU job failed or finished."""

    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__("scheduler_state_unknown; squeue/sacct could not confirm every job")


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
        diagnostics = []

        def query(command):
            try:
                result = subprocess.run(
                    command, check=False, capture_output=True, text=True, timeout=20
                )
                diagnostics.append(
                    {
                        "command": command,
                        "returncode": result.returncode,
                        "stdout": result.stdout[:4096],
                        "stderr": result.stderr[:4096],
                    }
                )
                return result
            except (OSError, subprocess.TimeoutExpired) as error:
                diagnostics.append(
                    {
                        "command": command,
                        "returncode": None,
                        "error": str(error),
                        "stdout": str(getattr(error, "stdout", "") or "")[:4096],
                        "stderr": str(getattr(error, "stderr", "") or "")[:4096],
                    }
                )
                return None

        queued = query(["squeue", "--noheader", "--jobs", ",".join(ids), "--format=%A"])
        if queued is not None and queued.returncode == 0:
            return set(queued.stdout.split())
        accounting = query(
            [
                "sacct",
                "--noheader",
                "--allocations",
                "--parsable2",
                "--jobs",
                ",".join(ids),
                "--format=JobID%64,State%32",
            ]
        )
        states = {}
        if accounting is not None and accounting.returncode == 0:
            for line in accounting.stdout.splitlines():
                fields = line.split("|")
                if len(fields) >= 2 and fields[0].strip() in ids and fields[1].strip():
                    states.setdefault(fields[0].strip(), set()).add(
                        fields[1].split()[0].rstrip("+")
                    )
        if set(states) != set(ids):
            raise SchedulerQueryError(diagnostics)
        terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"}
        return {job for job, values in states.items() if not values.issubset(terminal)}

    def evidence(self, ids):
        """Read registered job outcomes for diagnostics, never infer task completion."""
        command = [
            "sacct",
            "--noheader",
            "--allocations",
            "--parsable2",
            "--jobs",
            ",".join(ids),
            "--format=JobIDRaw,State,ExitCode,Reason",
        ]
        rows = {job: {"state": "UNKNOWN", "exit_code": None} for job in ids}
        try:
            result = subprocess.run(
                command, check=False, capture_output=True, text=True, timeout=20
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            return {job: {**row, "query_error": str(error)[:1024]} for job, row in rows.items()}
        if result.returncode:
            return {job: {**row, "query_error": result.stderr[:1024]} for job, row in rows.items()}
        for line in result.stdout.splitlines():
            cells = line.split("|")
            if len(cells) >= 4 and cells[0] in rows:
                rows[cells[0]] = {
                    "state": cells[1].split()[0].rstrip("+") if cells[1].strip() else "UNKNOWN",
                    "exit_code": cells[2] or None,
                    "reason": cells[3],
                }
        return rows


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
        prepared = runner.prepare(state, _defer_archive=True)
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
                if absent:
                    from phase_agent.tools.relax_mc_batch.failure_diagnostics import (
                        record_remote_diagnostics,
                    )

                    record_remote_diagnostics(
                        root,
                        state,
                        list(collected.values()),
                        absent,
                        getattr(runner, "last_collection_report", {}),
                        io,
                    )
                    checkpoint(root, state, manager)
                    raise RuntimeError(
                        "remote_results_unverified; see return_diagnostics.json; no resubmission"
                    )
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
    if any(r.get("status") in {"failed", "timeout", "cancelled"} for r in collected.values()):
        from phase_agent.tools.relax_mc_batch.failure_diagnostics import record_remote_diagnostics

        record_remote_diagnostics(
            root,
            state,
            list(collected.values()),
            [],
            getattr(runner, "last_collection_report", {}),
            io,
        )
    return state, list(collected.values())


def completion_outcome(state, status, reason=None):
    if status == "completed":
        from phase_agent.tools.relax_mc_batch.failure_diagnostics import unverified_task_ids

        if unverified_task_ids(state):
            return (
                "paused",
                "remote_results_unverified; inspect return_diagnostics.json; no resubmission",
            )
        if any(
            t.get("status") in {"failed", "timeout", "cancelled"} for t in state.get("tasks") or []
        ):
            return "partial", reason or "verified_task_failures_require_review; no resubmission"
    return status, reason


def publish(root, state, manifest, status, reason=None):
    root = Path(root)
    from phase_agent.tools.relax_mc_batch.failure_diagnostics import (
        record_remote_diagnostics,
        compact_diagnostics,
    )

    diagnostic = (state.get("relax_mc_remote") or {}).get("failure_diagnostics")
    if diagnostic is None:
        from phase_agent.tools.relax_mc_batch.failure_diagnostics import unverified_task_ids

        diagnostic = record_remote_diagnostics(
            root, state, state.get("tasks") or [], unverified_task_ids(state), {}, None
        )
    diagnostic = {
        **deepcopy(diagnostic),
        **{
            key: manifest[key]
            for key in ("bundle_id", "plan_checksum", "model_version", "config_version")
        },
    }
    if reason and (root / "failure.json").is_file():
        failure = read(root / "failure.json")
        diagnostic["controller"] = {
            "origin": "remote_controller",
            "error": reason[:2048],
            "traceback": str(failure.get("traceback") or "")[-4096:],
        }
    write(root / "return_diagnostics.json", diagnostic)
    write(
        root / "result_summary.json",
        {
            "bundle_id": manifest["bundle_id"],
            "plan_checksum": manifest["plan_checksum"],
            "model_version": manifest["model_version"],
            "config_version": manifest["config_version"],
            "status": status,
            "reason": reason,
            "diagnostics": compact_diagnostics(diagnostic, "return_diagnostics.json"),
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
    # Keep full remote checkpoints in place for resumption; publish only the
    # importer contract and scientifically referenced artifacts.
    from phase_agent.tools.relax_mc_batch.return_payload import compact_state, result_files
    import json

    candidates = candidate_pool(state)
    files = [
        root / name
        for name in (
            "workflow_state.json",
            "result_summary.json",
            "candidate_pool.json",
            "return_diagnostics.json",
        )
    ]
    if (root / "failure.json").is_file():
        files.append(root / "failure.json")
    if (root / "scheduler_diagnostics.json").is_file():
        files.append(root / "scheduler_diagnostics.json")
    files.extend(result_files(root, state, candidates))
    write(root / "candidate_pool.json", rebase(candidates, root, portable=True))
    payload = json.dumps(
        rebase(compact_state(state), root, portable=True),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return create_bundle(
        root, files, root / "results.zip", kind="results", payloads={"workflow_state.json": payload}
    )


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
                                    "tasks_in_progress",
                                }
                                else "completed"
                            )
                            reason = plan.get("reason", plan["status"])
                            break
                expected = {
                    t["task_id"]
                    for t in state["tasks"]
                    if t.get("status") in {"pending", "running"}
                }
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
                from phase_agent.tools.relax_mc_batch.failure_diagnostics import (
                    unverified_task_ids,
                    record_remote_diagnostics,
                )

                unverified = unverified_task_ids(state, expected)
                if unverified:
                    for batch in state.get("slurm_batches") or []:
                        if unverified.intersection(batch.get("task_ids") or []):
                            batch.pop("relax_mc_wave_done", None)
                    record_remote_diagnostics(root, state, results, unverified, {}, io)
                    status, reason = (
                        "paused",
                        "remote_results_unverified; inspect return_diagnostics.json; no resubmission",
                    )
                    checkpoint(root, state, manager)
                    break
                if any(row.get("status") != "completed" for row in results):
                    status, reason = (
                        "paused",
                        "Phase failures require inspection; no automatic resubmission",
                    )
                    checkpoint(root, state, manager)
                    break
                meta = state["relax_mc_remote"]
                if phase:
                    meta["completed_mc_waves"].append(phase - 1)
                meta["phase"] = phase + 1
                checkpoint(root, state, manager)
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
        status, reason = completion_outcome(state, status, reason)
        state["relax_mc_remote"]["status"] = status
        checkpoint(root, state, manager)
        archive = publish(root, state, manifest, status, reason)
        return {
            "status": status,
            "reason": reason,
            "archive": archive,
            "submitted": True,
        }

"""Build a verified recovery return from existing artifacts; never submits or computes."""

from copy import deepcopy
from pathlib import Path
import hashlib
import shutil
from uuid import uuid4
from filelock import FileLock

BUILD_VERSION = "collect-only-v2"

from phase_agent.tools.remote.transfer_bundle import (
    import_bundle,
    create_bundle,
    local_path,
    validate_bundle,
)
from phase_agent.tools.remote.integrity import verify_result
from phase_agent.tools.remote.resolve_local_relax_structure import resolve_local_relax_structure
from phase_agent.tools.state.normalize_task_result import normalize_task_result
from phase_agent.persistence.ledger.result_validation import validate_scientific_result
from phase_agent.tools.relax_mc_batch.files import read, write, rebase
from phase_agent.tools.relax_mc_batch.analysis import candidate_pool


def build_recovery_return(
    original_archive, evidence_directory, destination, *, job_evidence, remote_root
):
    destination = local_path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination / ".collect-only.lock"), timeout=10):
        return _build_recovery_return(
            original_archive,
            evidence_directory,
            destination,
            job_evidence=job_evidence,
            remote_root=remote_root,
        )


def _build_recovery_return(
    original_archive, evidence_directory, destination, *, job_evidence, remote_root
):
    """Only supplement tasks already registered in the original return's frozen scope."""
    destination, evidence = (
        local_path(destination).resolve(),
        local_path(evidence_directory).resolve(),
    )
    if (
        destination == evidence
        or destination.is_relative_to(evidence)
        or evidence.is_relative_to(destination)
    ):
        raise ValueError("recovery_destination_must_be_separate")
    snapshot_path = evidence / "workflow_state.json"
    if not snapshot_path.is_file():
        raise ValueError("recovery_submission_snapshot_required")
    submission = read(snapshot_path)
    matches = [
        b
        for b in submission.get("slurm_batches") or []
        if str(b.get("job_id")) == str(job_evidence.get("job_id"))
    ]
    requested_ids = set(job_evidence.get("task_ids") or [])
    if (
        not matches
        or not requested_ids
        or not requested_ids <= {tid for b in matches for tid in b.get("task_ids") or []}
    ):
        raise ValueError("recovery_job_task_submission_mismatch")
    original_digest = hashlib.sha256(Path(original_archive).read_bytes()).hexdigest()
    fingerprint = hashlib.sha256((BUILD_VERSION + original_digest).encode())
    import json

    fingerprint.update(json.dumps(job_evidence, sort_keys=True).encode())
    fingerprint.update(hashlib.sha256(snapshot_path.read_bytes()).digest())
    for item in sorted((evidence / "jobs").rglob("*")):
        if item.is_symlink():
            raise ValueError("recovery_symlink_not_allowed")
        if item.is_file():
            fingerprint.update(str(item.relative_to(evidence)).encode())
            fingerprint.update(hashlib.sha256(item.read_bytes()).digest())
    identity = fingerprint.hexdigest()
    root = destination / identity[:16]
    archive = root / "recovered.results.zip"
    if archive.is_file() and (root / "recovery_evidence.json").is_file():
        validate_bundle(archive, extra_result_suffixes=(".csv", ".bin"))
        report = read(root / "recovery_evidence.json")
        if report.get("build_identity") != identity:
            raise ValueError("recovery_existing_build_identity_mismatch")
        return {
            "status": read(root / "result_summary.json")["status"],
            "archive": str(archive),
            "recovery": report,
            "submitted": False,
            "idempotent_replay": True,
        }
    if root.exists():
        interrupted = destination / ("interrupted-" + identity[:16] + "-" + uuid4().hex)
        if not root.resolve().is_relative_to(
            destination
        ) or not interrupted.resolve().is_relative_to(destination):
            raise ValueError("recovery_staging_outside_destination")
        root.rename(interrupted)
    import_bundle(original_archive, root, extra_result_suffixes=(".csv", ".bin"))
    state = rebase(read(root / "workflow_state.json"), root)
    summary = read(root / "result_summary.json")
    original_digest = hashlib.sha256(Path(original_archive).read_bytes()).hexdigest()
    bindings = {
        "model_version": summary["model_version"],
        "config_version": summary["config_version"],
    }
    tasks = {t["task_id"]: t for t in state.get("tasks") or []}
    requested = set(job_evidence.get("task_ids") or [])
    if not requested or not requested <= tasks.keys() or not job_evidence.get("job_id"):
        raise ValueError("recovery_requires_registered_job_task_scope")
    if job_evidence.get("state") not in {
        "COMPLETED",
        "FAILED",
        "TIMEOUT",
        "CANCELLED",
        "OUT_OF_MEMORY",
        "NODE_FAIL",
    }:
        raise ValueError("recovery_scheduler_state_unknown; collect new terminal job evidence")
    if not remote_root.startswith("/") or remote_root == "/":
        raise ValueError("recovery_remote_root_required")

    def relocate(value):
        if isinstance(value, dict):
            return {k: relocate(v) for k, v in value.items()}
        if isinstance(value, list):
            return [relocate(v) for v in value]
        if isinstance(value, str) and value.startswith(remote_root.rstrip("/") + "/"):
            relative = value[len(remote_root.rstrip("/")) + 1 :]
            path = (root / relative).resolve()
            if not path.is_relative_to(root):
                raise ValueError("recovery_path_outside_batch")
            return str(path)
        return value

    recovered, missing = [], []
    # Copy only paths belonging to the registered tasks; signed JSON bytes stay unchanged.
    for tid in sorted(requested):
        task = tasks[tid]
        if any(task.get(k) != v for k, v in bindings.items()):
            raise ValueError("recovery_task_binding_mismatch:" + tid)
        relative = Path(task["result_path"]).relative_to(root)
        source = evidence / relative
        if not source.is_file() or not source.with_name("task.finished.json").is_file():
            missing.append(tid)
            continue
        if not source.resolve().is_relative_to(evidence):
            raise ValueError("recovery_evidence_path_outside_root")
        checked = verify_result(source, source.with_name("task.finished.json"), task)
        if not checked["valid"]:
            raise ValueError("recovery_result_integrity_failed:" + tid + ":" + checked["reason"])
        worker = Path(task["input_path"]).parent.relative_to(root)
        for folder in (source.parent, evidence / worker):
            if not folder.is_dir():
                continue
            for item in folder.rglob("*"):
                if item.is_symlink():
                    raise ValueError("recovery_symlink_not_allowed")
                if not item.is_file():
                    continue
                rel = item.relative_to(evidence)
                if item.suffix.lower() not in {".json", ".gz", ".bin", ".vasp", ".poscar"}:
                    continue
                target = root / rel
                if target.exists() and target.read_bytes() != item.read_bytes():
                    raise ValueError("recovery_existing_artifact_conflict:" + str(rel))
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)
        result_path = root / relative
        payload = resolve_local_relax_structure(checked["result"], result_path.parent)
        if checked["result"].get("status") == "completed" and payload is None:
            raise ValueError("recovery_final_structure_missing:" + tid)
        normalized = normalize_task_result(
            {**task, "object_id": task.get("object_id", task["structure_id"])},
            payload or checked["result"],
        )
        validate_scientific_result(normalized)
        normalized = relocate(normalized)
        normalized["result_path"] = str(result_path)
        tasks[tid] = normalized
        recovered.append(tid)
    state["tasks"] = [tasks[t["task_id"]] for t in state["tasks"]]
    remaining = [
        t["task_id"]
        for t in state["tasks"]
        if t.get("status") not in {"completed", "failed", "cancelled", "timeout"}
    ]
    report = {
        "schema_version": 1,
        "build_identity": identity,
        "builder_version": BUILD_VERSION,
        "kind": "collect_only",
        "original_archive_checksum": original_digest,
        "job_evidence": deepcopy(job_evidence),
        "recovered_task_ids": recovered,
        "unverified_task_ids": sorted(set(missing) | set(remaining)),
        "submitted": False,
    }
    phase = (submission.get("relax_mc_remote") or {}).get("phase")
    failed = [
        t["task_id"]
        for t in state["tasks"]
        if t.get("status") in {"failed", "timeout", "cancelled"}
    ]
    segments = {
        t["segment_index"]
        for t in state["tasks"]
        if t.get("stage") == "deep_search" and type(t.get("segment_index")) is int
    }
    complete = (
        not remaining and not failed and phase == 3 and (not segments or {0, 1, 2} <= segments)
    )
    report["batch_complete"] = complete
    report["verified_failed_task_ids"] = failed
    report["next_phase_not_submitted"] = not remaining and not failed and not complete
    summary["failed_task_ids"] = failed
    summary.update(
        status="completed" if complete else "partial",
        reason=None
        if complete
        else "verified_task_failures_require_review; no jobs submitted"
        if failed
        else "current_phase_collected; next phase not submitted"
        if not remaining
        else "recovery_results_incomplete; no jobs submitted",
        recovery=report,
    )
    waves = list(summary.get("completed_mc_waves") or [])
    if segments:
        waves = [
            index
            for index in sorted(segments)
            if all(
                t.get("status") == "completed"
                for t in state["tasks"]
                if t.get("stage") == "deep_search" and t.get("segment_index") == index
            )
        ]
    elif (
        not remaining
        and not failed
        and type(phase) is int
        and 1 <= phase <= 3
        and phase - 1 not in waves
    ):
        waves.append(phase - 1)
    summary["completed_mc_waves"] = sorted(waves)
    candidates = candidate_pool(state)
    write(root / "workflow_state.json", rebase(state, root, portable=True))
    write(root / "candidate_pool.json", rebase(candidates, root, portable=True))
    write(root / "result_summary.json", summary)
    write(root / "recovery_evidence.json", report)
    files = [
        f
        for f in root.rglob("*")
        if f.is_file()
        and (
            f.is_relative_to(root / "jobs")
            or f.name
            in {
                "workflow_state.json",
                "candidate_pool.json",
                "result_summary.json",
                "recovery_evidence.json",
            }
        )
    ]
    archive = create_bundle(root, files, root / "recovered.results.zip", kind="results")
    return {
        "status": summary["status"],
        "archive": str(archive),
        "recovery": report,
        "submitted": False,
    }

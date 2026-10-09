"""Local-master orchestration for remote, offline calculations."""

from copy import deepcopy
from pathlib import Path, PurePosixPath

from execution_layer.step_runner.file_protocol import read_json, write_json


def _batch_locations(batch, local_batch_root, remote_batch_root):
    local_root = Path(local_batch_root).resolve()
    local = Path(batch.get("upload_directory") or (local_root / batch["batch_id"])).resolve()
    try:
        relative = local.relative_to(local_root)
    except ValueError as exc:
        raise ValueError(f"batch upload_directory escaped local root: {local}") from exc
    remote = PurePosixPath(remote_batch_root).joinpath(*relative.parts)
    return local, remote


def sync_results(state_path, *, local_batch_root, remote_batch_root, transport):
    """Download artifacts only; reconciliation remains a local operation."""
    state = read_json(state_path, {}) or {}; synced = []
    result_locations = {}
    for batch in state.get("slurm_batches") or []:
        if batch.get("status") not in {"submitted", "running", "completed", "results_synced"}:
            continue
        local, remote = _batch_locations(batch, local_batch_root, remote_batch_root)
        local_results = Path(batch.get("results_directory") or (local.parent / "results"))
        remote_batch = PurePosixPath(batch.get("remote_path") or remote)
        remote_results = (remote_batch.parent.parent if remote_batch.parent.name == "inputs"
                          else remote_batch.parent) / "results"
        key = (str(local_results.resolve()), str(remote_results))
        result_locations.setdefault(key, []).append(batch["batch_id"])
    for (local_results, remote_results), batch_ids in result_locations.items():
        result = transport.download_tree(remote_results, local_results)
        if result.get("status") == "synced":
            synced.extend(batch_ids)
    return {"status": "synced" if synced else "nothing_to_sync", "batch_ids": synced}


def prepare_local(state_path, *, prepare_callback):
    """Run local recovery/Agent/analysis and persist the returned master state."""
    state = read_json(state_path, {}) or {}
    result = prepare_callback(deepcopy(state))
    updated = result.get("state") if isinstance(result, dict) else None
    if not isinstance(updated, dict):
        raise TypeError("prepare_callback must return a dict containing state")
    write_json(state_path, updated)
    return result


def sync_tasks(state_path, *, local_batch_root, remote_batch_root, transport):
    """Upload immutable snapshots and update only the local master ledger."""
    state = read_json(state_path, {}) or {}; synced = []
    for batch in state.get("slurm_batches") or []:
        if batch.get("status") not in {"prepared", "synced"}: continue
        local, remote = _batch_locations(batch, local_batch_root, remote_batch_root)
        outcome = transport.upload_tree(local, str(remote))
        if outcome.get("status") not in {"synced", "already_synced"}:
            raise RuntimeError(f"batch sync failed: {batch['batch_id']}: {outcome}")
        batch.update({"status": "synced", "remote_path": str(remote)})
        synced.append(batch["batch_id"])
    write_json(state_path, state)
    return {"status": "synced" if synced else "nothing_to_sync", "state": state,
            "batch_ids": synced}


def submit_remote(state_path, *, scheduler):
    """Submit once; an uncertain previous attempt is queried before retry."""
    state = read_json(state_path, {}) or {}; submitted = []
    for batch in state.get("slurm_batches") or []:
        token = batch.setdefault("submission_token", f"submit:{batch['batch_id']}:{batch.get('checksum')}")
        if batch.get("status") == "submission_uncertain":
            found = scheduler.query(job_id=batch.get("job_id"), submission_token=token)
            if found.get("job_id"):
                batch.update({"status": found.get("status", "submitted"), "job_id": found["job_id"]})
            else: continue
        if batch.get("status") != "synced" or batch.get("job_id"): continue
        batch["status"] = "submitting"; write_json(state_path, state)
        try:
            result = scheduler.submit(
                script_path=str(PurePosixPath(batch["remote_path"]) / "submit.sbatch"),
                submission_token=token)
        except Exception:
            batch["status"] = "submission_uncertain"; write_json(state_path, state); raise
        if not result.get("job_id"):
            batch["status"] = "submission_uncertain"; write_json(state_path, state); continue
        batch.update({"status": result.get("status", "submitted"), "job_id": result["job_id"]})
        submitted.append({"batch_id": batch["batch_id"], "job_id": result["job_id"]})
        write_json(state_path, state)
    return {"status": "submitted" if submitted else "nothing_to_submit", "state": state,
            "submitted": submitted}


def query_remote(state_path, *, scheduler):
    state = read_json(state_path, {}) or {}; queried = []
    for batch in state.get("slurm_batches") or []:
        if not batch.get("job_id") or batch.get("status") in {"results_synced", "cancelled"}: continue
        result = scheduler.query(job_id=batch["job_id"], submission_token=batch.get("submission_token"))
        if result.get("status") not in {None, "not_found", "not_configured"}:
            batch["status"] = result["status"]
        queried.append({"batch_id": batch["batch_id"], **result})
    write_json(state_path, state)
    return {"status": "queried" if queried else "nothing_to_query", "state": state, "jobs": queried}


def resume(state_path):
    """Restore the local master ledger and report safe next operations."""
    state = read_json(state_path)
    if state is None: raise FileNotFoundError(state_path)
    statuses = {row.get("status") for row in state.get("slurm_batches") or []}; steps = []
    if statuses & {"submitted", "running", "completed"}: steps += ["query_remote", "sync_results", "prepare_local"]
    if "prepared" in statuses: steps.append("sync_tasks")
    if "synced" in statuses: steps.append("submit_remote")
    if not steps: steps.append("prepare_local")
    return {"status": "resumed", "state": state, "next_steps": list(dict.fromkeys(steps))}

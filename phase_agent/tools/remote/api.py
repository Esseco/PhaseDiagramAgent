"""Unified local-Agent / remote-compute interface."""

from phase_agent.tools.remote.portable_batch_runner import RemoteBatchRunner
from phase_agent.tools.remote.scheduler import MockRemoteScheduler, RemoteSchedulerAdapter
from phase_agent.tools.remote.transport import CommandTransferAdapter, LocalMirrorTransport
from phase_agent.tools.remote.workflow import (
    prepare_local,
    query_remote,
    resume,
    submit_remote,
    sync_results,
    sync_tasks as _sync_tasks,
)
from phase_agent.tools.step_runner.file_protocol import read_json, write_json

SECRET_NAMES = {"api_key", "token", "password", "secret", "private_key"}


def sync_tasks(state_path, **kwargs):
    state = read_json(state_path, {}) or {}
    _reject_secrets(state.get("tasks") or [])
    return _sync_tasks(state_path, **kwargs)


def cancel_remote(state_path, *, scheduler, batch_id=None):
    state = read_json(state_path, {}) or {}
    cancelled = []
    for batch in state.get("slurm_batches") or []:
        if batch_id and batch.get("batch_id") != batch_id:
            continue
        if not batch.get("job_id") or batch.get("status") in {"cancelled", "results_synced"}:
            continue
        result = scheduler.cancel(job_id=batch["job_id"])
        if result.get("status") == "cancelled":
            batch["status"] = "cancelled"
            cancelled.append(batch["batch_id"])
    write_json(state_path, state)
    return {
        "status": "cancelled" if cancelled else "nothing_to_cancel",
        "state": state,
        "batch_ids": cancelled,
    }


def _reject_secrets(value, path="tasks"):
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in SECRET_NAMES and item not in {None, ""}:
                raise ValueError(f"secret field must not be synchronized: {path}.{key}")
            _reject_secrets(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_secrets(item, f"{path}[{index}]")


__all__ = [
    "sync_results",
    "prepare_local",
    "sync_tasks",
    "submit_remote",
    "query_remote",
    "cancel_remote",
    "resume",
    "RemoteBatchRunner",
    "CommandTransferAdapter",
    "LocalMirrorTransport",
    "RemoteSchedulerAdapter",
    "MockRemoteScheduler",
]

"""No-network example for the local-Agent / remote-compute file protocol."""

from pathlib import Path
from phase_agent.tools.remote.api import RemoteBatchRunner, submit_remote, sync_tasks
from phase_agent.tools.remote.verified_mock import (
    FileBackedMockRemoteScheduler,
    VerifiedLocalMirrorTransport,
)
from phase_agent.tools.step_runner.file_protocol import write_json


def build_dry_run(root="outputs/remote-dry-run"):
    root = Path(root)
    local = root / "local-batches"
    remote = root / "mock-remote/batches"
    task = {
        "task_id": "DRY-T1",
        "task_key": "DRY-K1",
        "structure_id": "S1",
        "object_id": "S1",
        "stage": "deep_search",
        "status": "pending",
        "parameters": {},
    }
    state = {
        "confirmed_config_version": "dry-config",
        "active_model_version": "dry-model",
        "tasks": [task],
        "pending_tasks": [task],
        "budget_reservations": {
            "DRY-K1": {"status": "reserved", "reserved_cost": 1, "stage": "deep_search"}
        },
    }
    worker = [
        "python3",
        "-m",
        "phase_agent.tools.remote.worker_cli",
        "--executor",
        "phase_agent.science.mlip.slurm_executor:execute_mlip_task",
    ]
    runner = RemoteBatchRunner(local, worker_command=worker)
    state = runner.prepare(state)["state"]
    state_path = root / "local-state.json"
    write_json(state_path, state)
    transport = VerifiedLocalMirrorTransport()
    scheduler = FileBackedMockRemoteScheduler(root / "mock-remote/job-status")
    synced = sync_tasks(
        state_path, local_batch_root=local, remote_batch_root=remote, transport=transport
    )
    submitted = submit_remote(state_path, scheduler=scheduler)
    return {"state_path": str(state_path), "sync": synced["status"], "submit": submitted["status"]}


if __name__ == "__main__":
    import json

    print(json.dumps(build_dry_run(), ensure_ascii=False, indent=2))

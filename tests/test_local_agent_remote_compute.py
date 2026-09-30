import json
from pathlib import Path
import pytest

from execution_layer.remote.api import (RemoteBatchRunner, cancel_remote, query_remote, resume,
    submit_remote, sync_results, sync_tasks)
from execution_layer.remote.worker import run_remote_task
from execution_layer.remote.verified_mock import FileBackedMockRemoteScheduler, VerifiedLocalMirrorTransport
from execution_layer.step_runner.file_protocol import write_json


def _task(index, stage):
    return {"task_id": f"T{index:04d}", "task_key": f"K{index:04d}", "structure_id": f"S{index:04d}",
            "object_id": f"S{index:04d}", "stage": stage, "status": "pending", "parameters": {}}


def _state(tasks):
    return {"confirmed_config_version": "cfg-1", "active_model_version": "m1", "tasks": tasks,
            "pending_tasks": list(tasks), "budget_reservations": {row["task_key"]:
            {"status": "reserved", "reserved_cost": 1, "stage": row["stage"]} for row in tasks}}


def test_remote_batch_sizes_versions_and_portable_paths(tmp_path):
    tasks = [_task(i, "relax_and_feature") for i in range(101)] + [_task(i + 200, "deep_search") for i in range(21)]
    state = _state(tasks); runner = RemoteBatchRunner(tmp_path / "local", worker_command=["worker"])
    sizes = []
    while True:
        output = runner.prepare(state); state = output["state"]
        if output["status"] == "no_tasks": break
        manifest = json.loads(Path(output["batch"]["manifest_path"]).read_text(encoding="utf-8"))
        sizes.append(len(manifest)); assert all(not Path(row["input_path"]).is_absolute() for row in manifest)
        assert all(row["config_version"] == "cfg-1" and row["model_version"] == "m1" for row in manifest)
        assert output["batch"]["checksum"]
    assert sizes == [100, 1, 10, 10, 1]


def test_sync_submit_query_worker_recovery_and_repeat_are_safe(tmp_path):
    local, remote, state_path = tmp_path / "local", tmp_path / "remote", tmp_path / "state.json"
    runner = RemoteBatchRunner(local, worker_command=["worker"])
    state = runner.prepare(_state([_task(1, "deep_search")]))["state"]; write_json(state_path, state)
    transport = VerifiedLocalMirrorTransport(); scheduler = FileBackedMockRemoteScheduler(remote / "job-status")
    assert sync_tasks(state_path, local_batch_root=local, remote_batch_root=remote / "batches", transport=transport)["status"] == "synced"
    assert sync_tasks(state_path, local_batch_root=local, remote_batch_root=remote / "batches", transport=transport)["status"] == "synced"
    assert len(submit_remote(state_path, scheduler=scheduler)["submitted"]) == 1
    assert submit_remote(state_path, scheduler=scheduler)["status"] == "nothing_to_submit"
    assert scheduler.submit_calls == 1; assert query_remote(state_path, scheduler=scheduler)["status"] == "queried"
    saved = json.loads(state_path.read_text())
    remote_batch = Path(saved["slurm_batches"][0]["remote_path"])
    remote_manifest = remote_batch / "manifest.json"
    def complete_with_structure(task):
        final = Path(task["calculation_directory"]) / "final.vasp"
        final.write_text("mock final structure", encoding="utf-8")
        return {"status": "completed", "actual_cost": .5,
                "outputs": {"structure_path": final.name}}

    run_remote_task(remote_manifest, 0, executor=complete_with_structure)
    # large trajectories remain remote
    trajectory = remote_batch / "00000-T0001/large.traj"; trajectory.write_text("large")
    sync_results(state_path, local_batch_root=local, remote_batch_root=remote / "batches", transport=transport)
    collection = runner.collect_results_with_report(json.loads(state_path.read_text()))
    assert collection["results"], collection["report"]
    assert collection["results"][0]["actual_cost"] == .5
    local_batch = Path(saved["slurm_batches"][0]["upload_directory"])
    assert not (local_batch / "00000-T0001/large.traj").exists()
    assert "query_remote" in resume(state_path)["next_steps"]
    assert cancel_remote(state_path, scheduler=scheduler)["status"] == "cancelled"


def test_secret_is_not_uploaded(tmp_path):
    state = _state([_task(1, "deep_search")]); state["tasks"][0]["parameters"]["api_key"] = "no-upload"
    runner = RemoteBatchRunner(tmp_path / "local", worker_command=["worker"]); state = runner.prepare(state)["state"]
    path = tmp_path / "state.json"; write_json(path, state)
    with pytest.raises(ValueError, match="secret field"):
        sync_tasks(path, local_batch_root=tmp_path / "local", remote_batch_root=tmp_path / "remote",
                   transport=VerifiedLocalMirrorTransport())


def test_worker_does_not_publish_completion_when_final_structure_is_missing(tmp_path):
    runner = RemoteBatchRunner(tmp_path / "local", worker_command=["worker"])
    output = runner.prepare(_state([_task(1, "deep_search")]))
    manifest_path = Path(output["batch"]["manifest_path"])
    entry = json.loads(manifest_path.read_text(encoding="utf-8"))[0]
    with pytest.raises(FileNotFoundError):
        run_remote_task(manifest_path, 0, executor=lambda task: {
            "status": "completed", "outputs": {"structure_path": "final.vasp"}})
    published = (manifest_path.parent / entry["result_path"]).resolve()
    assert not published.with_name("task.finished.json").exists()

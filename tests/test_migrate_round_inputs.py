import json
from phase_agent.tools.local.migrate_round_inputs import migrate


def test_offline_migration_preserves_results_remote_paths_and_task_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr("phase_agent.tools.local.migrate_round_inputs.require_agent_offline", lambda port: None)
    batch = tmp_path / "submissions/epoch0_m1/Search-group-0001/Relax-0001/batch"
    batch.mkdir(parents=True)
    (batch / "task.json").write_bytes(b"immutable task")
    results = batch.parent / "results"
    results.mkdir()
    (results / "result.json").write_bytes(b"unchanged result")
    training = tmp_path / "submissions/epoch0_m1/MLIP-finetune-round-0001"
    training.mkdir()
    (training / "GPU.sh").write_bytes(b"same gpu")
    (training / "collect_training_results.py").write_text('output = root / "results"')
    path = tmp_path / "workflow_state/state.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"slurm_batches": [{"upload_directory": str(batch), "remote_directory": "/hpc/batch"}],
        "tasks": [{"input_path": str(batch / "task.json"), "result_path": str(results / "result.json")}],
        "remote_finetune_jobs": {"j": {"directory": str(training)}},
        "confirmed_config": {"old_path": str(batch)}}))
    before = path.read_bytes()
    assert migrate(tmp_path)["moves"]
    assert path.read_bytes() == before
    report = migrate(tmp_path, apply=True)
    updated = json.loads(path.read_text())
    assert (batch.parent / "inputs/batch/task.json").read_bytes() == b"immutable task"
    assert (results / "result.json").read_bytes() == b"unchanged result"
    assert (training / "inputs/GPU.sh").read_bytes() == b"same gpu"
    assert updated["slurm_batches"][0]["remote_directory"] == "/hpc/batch"
    assert updated["confirmed_config"]["old_path"] == str(batch)
    assert updated["tasks"][0]["result_path"] == str(results / "result.json")
    assert report["backup"]
    assert not migrate(tmp_path)["moves"]

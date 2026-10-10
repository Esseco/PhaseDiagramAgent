import json
from phase_agent.tools.remote.build_upload_batch_directory import build_upload_batch_directory
from phase_agent.tools.local.migrate_epoch_upload_layout import migrate


def test_epoch_naming_preserves_ids(tmp_path):
    state = {}
    path, metadata = build_upload_batch_directory(tmp_path, state, batch_id="remote-000006",
        stage="deep_search", model_version="m1", operation_id="op", search_group_index=1,
        segment_index=1, parent_relax_round=1)
    assert path.parts[-5] == "epoch0_m1"
    assert path.parent.name == "inputs"
    assert path.parts[-3] == "Relax-0001_MC-round-0002"
    assert path.name.endswith("remote-000006")


def test_epoch_migration_rewrites_local_paths_and_keeps_remote(tmp_path):
    current = tmp_path / "current"
    current.mkdir()
    source = tmp_path / "upload_batches/MLIP-round-0001_m1/Search-group-0001"
    source.mkdir(parents=True)
    state = {"slurm_batches": [{"status": "prepared", "upload_directory": str(source)}]}
    (current / "state.json").write_text(json.dumps(state))
    (source / "task.json").write_text(json.dumps({"local": str(source / "input"), "remote": "/remote/MLIP-round-0001_m1/input"}))
    result = migrate(tmp_path)
    target = tmp_path / "upload_batches/epoch0_m1/Search-group-0001"
    assert target.is_dir() and not source.exists()
    assert json.loads((current / "state.json").read_text())["slurm_batches"][0]["upload_directory"] == str(target)
    assert json.loads((target / "task.json").read_text())["remote"] == "/remote/MLIP-round-0001_m1/input"
    assert result["backup"]

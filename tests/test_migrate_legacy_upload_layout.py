import json
from pathlib import Path

from execution_layer.remote.migrate_legacy_upload_layout import migrate_legacy_upload_layout


def test_migrates_flat_batch_and_rewrites_saved_paths(tmp_path):
    root = tmp_path / "upload_batches"
    source = root / "remote-000001"
    task_dir = source / "00000-RELAX-1"
    task_dir.mkdir(parents=True)
    manifest = [{"batch_id": "remote-000001", "task_id": "RELAX-1",
                 "stage": "relax_and_feature", "model_version": "mace-mh-1",
                 "input_path": "00000-RELAX-1/task.json"}]
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (task_dir / "task.json").write_text("{}", encoding="utf-8")
    state = {"active_model_version": "mace-mh-1", "tasks": [{
        "task_id": "RELAX-1", "branch_id": "B-1", "stage": "relax_and_feature",
        "status": "pending", "batch_id": "remote-000001",
        "input_path": str(task_dir / "task.json"),
        "result_path": str(task_dir / "result.json")}],
        "slurm_batches": [{"batch_id": "remote-000001", "status": "prepared",
            "model_version": "mace-mh-1", "upload_directory": str(source),
            "manifest_path": str(source / "manifest.json"), "job_id": None}]}
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    result = migrate_legacy_upload_layout(state_path, root)

    assert result["status"] == "migrated" and result["batch_count"] == 1
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    batch = saved["slurm_batches"][0]
    target = Path(batch["upload_directory"])
    assert target.parts[-3:] == ("MLIP-round-0001_mace-mh-1", "Relax-screening",
                                 "Relax-submission-0001_remote-000001")
    assert target.is_dir() and not source.exists()
    assert Path(saved["tasks"][0]["input_path"]).is_file()
    assert Path(result["backup_path"]).is_file()
    assert migrate_legacy_upload_layout(state_path, root)["status"] == "already_migrated"


def test_dry_run_does_not_move_or_rewrite(tmp_path):
    root = tmp_path / "upload_batches"
    source = root / "remote-000001"
    source.mkdir(parents=True)
    (source / "manifest.json").write_text(json.dumps([{
        "stage": "deep_search", "model_version": "m1"}]), encoding="utf-8")
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"slurm_batches": [{
        "batch_id": "remote-000001", "status": "prepared", "model_version": "m1",
        "upload_directory": str(source)}]}), encoding="utf-8")
    before = state_path.read_bytes()
    result = migrate_legacy_upload_layout(state_path, root, dry_run=True)
    assert result["status"] == "planned"
    assert source.is_dir() and state_path.read_bytes() == before

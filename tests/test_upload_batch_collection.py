import json

from execution_layer.remote.batch_runner import RemoteBatchRunner
from execution_layer.remote.integrity import file_checksum


def test_collects_returned_results_from_existing_upload_directory_without_copy(tmp_path):
    upload_root = tmp_path / "upload_batches" / "remote-000007"
    task_dir = upload_root / "00000-RELAX-T1"
    task_dir.mkdir(parents=True)
    task = {
        "task_id": "RELAX-T1",
        "task_key": "relax-key-1",
        "batch_id": "remote-000007",
        "config_version": "config-v1",
        "model_version": "mace-mh-1",
        "task_checksum": "input-checksum-1",
        "protocol_version": 1,
        "input_file_version": "task-json-v1",
    }
    (task_dir / "task.json").write_text(json.dumps(task), encoding="utf-8")
    final_structure = task_dir / "initial_relaxed.vasp"
    final_structure.write_text("relaxed structure", encoding="utf-8")
    result = {
        **task,
        "status": "completed",
        "stage": "relax_and_feature",
        "outputs": {
            "structure_path": final_structure.name,
            "structure_checksum": file_checksum(final_structure),
            "energy": -12.5,
            "energy_unit": "eV",
            "relax_stopped_normally": True,
            "composition": {"Na": 1, "Fe": 1, "O": 2},
            "forces_ev_per_angstrom": [[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]],
        },
    }
    result_path = task_dir / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    marker = {
        **{key: task[key] for key in (
            "task_id", "task_key", "batch_id", "config_version", "model_version",
            "task_checksum", "protocol_version", "input_file_version",
        )},
        "status": "completed",
        "result_checksum": file_checksum(result_path),
    }
    (task_dir / "task.finished.json").write_text(json.dumps(marker), encoding="utf-8")

    # Deliberately omit task-level file paths: recovery must locate by batch_id/task_id.
    state = {
        "tasks": [{
            "task_id": task["task_id"], "task_key": task["task_key"],
            "batch_id": task["batch_id"], "stage": "relax_and_feature", "status": "pending",
        }],
        "slurm_batches": [{"batch_id": task["batch_id"], "upload_directory": str(upload_root)}],
    }
    runner = RemoteBatchRunner(tmp_path / "unused-output", worker_command=[])
    collected = runner.collect_results_with_report(state)

    assert collected["report"]["status"] == "ready"
    assert collected["report"]["recovered_count"] == 1
    recovered = collected["results"][0]
    assert recovered["outputs"]["energy"] == -12.5
    assert recovered["outputs"]["structure_path"] == str(final_structure.resolve())
    assert recovered["calculation_result_path"] == str(result_path.resolve())
    assert "forces_ev_per_angstrom" not in recovered["outputs"]
    assert recovered["outputs"]["forces_stored_in_result_file"] is True
    assert json.loads(result_path.read_text(encoding="utf-8"))["outputs"]["forces_ev_per_angstrom"] == [
        [0.1, 0.2, 0.3], [0.0, 0.0, 0.0]
    ]
    assert final_structure.read_text(encoding="utf-8") == "relaxed structure"
    assert not (tmp_path / "unused-output" / task["batch_id"]).exists()


def test_collection_reports_missing_returned_files_without_mutating_upload_folder(tmp_path):
    upload_root = tmp_path / "upload_batches" / "remote-000008"
    task_dir = upload_root / "00000-RELAX-T2"
    task_dir.mkdir(parents=True)
    task = {"task_id": "RELAX-T2", "task_key": "relax-key-2", "batch_id": "remote-000008"}
    (task_dir / "task.json").write_text(json.dumps(task), encoding="utf-8")
    state = {
        "tasks": [{**task, "stage": "relax_and_feature", "status": "pending"}],
        "slurm_batches": [{"batch_id": task["batch_id"], "upload_directory": str(upload_root)}],
    }

    report = RemoteBatchRunner(tmp_path / "unused-output", worker_command=[]).collect_results_with_report(state)["report"]

    assert report["status"] == "waiting"
    assert report["missing_result_count"] == 1
    assert report["recovered_count"] == 0
    assert sorted(path.name for path in task_dir.iterdir()) == ["task.json"]

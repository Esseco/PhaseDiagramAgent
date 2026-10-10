import json
from phase_agent.tools.remote.export_batch_result import export_batch_result
from phase_agent.tools.remote.integrity import file_checksum


def test_logs_and_traces_stay_remote_but_resume_checkpoint_is_preserved(tmp_path):
    task = tmp_path / "batch" / "00000-MC-A"
    task.mkdir(parents=True)
    result = {"task_id": "A", "task_key": "key", "stage": "deep_search", "status": "failed", "outputs": {}}
    path = task / "result.json"
    path.write_text(json.dumps(result))
    (task / "task.finished.json").write_text(json.dumps({"task_id": "A", "task_key": "key",
        "status": "failed", "result_checksum": file_checksum(path)}))
    for name in ("task.stdout.log", "trace.json.gz", "checkpoint.json"):
        (task / name).write_bytes(b"retained remote evidence")
    destination = export_batch_result(task, tmp_path / "results")
    assert not (destination / "task.stdout.log").exists()
    assert not (destination / "trace.json.gz").exists()
    assert (destination / "checkpoint.json").exists()
    manifest = json.loads((destination / "remote_artifacts.json").read_text())
    assert len(manifest["artifacts"]) == 2
    assert all(row["size_bytes"] > 0 for row in manifest["artifacts"])
    assert (task / "task.stdout.log").is_file()
    marker = json.loads((destination / "task.finished.json").read_text())
    assert marker["result_checksum"] == file_checksum(destination / "result.json")

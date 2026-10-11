import json
import pytest
from phase_agent.tools.remote.export_batch_result import export_batch_result
from phase_agent.tools.remote.integrity import file_checksum
from phase_agent.tools.remote.transfer_bundle import import_bundle


def terminal_task(root, status):
    root.mkdir(parents=True)
    payload = {
        "task_id": "A",
        "task_key": "key",
        "stage": "deep_search",
        "status": status,
        "outputs": {},
    }
    (root / "result.json").write_text(json.dumps(payload))
    (root / "task.finished.json").write_text(
        json.dumps({**payload, "result_checksum": file_checksum(root / "result.json")})
    )
    (root / "checkpoint.json").write_text("{}")
    return root


@pytest.mark.parametrize("status", ["completed", "failed", "timeout", "cancelled"])
def test_terminal_export_automatically_creates_downloadable_archive(tmp_path, status):
    task = terminal_task(tmp_path / "inputs/batch/00000-MC-A", status)
    results = tmp_path / "results"
    destination = export_batch_result(task, results)
    archive = results / f"{destination.name}.results.zip"
    assert archive.is_file()
    downloaded = tmp_path / "downloaded"
    import_bundle(archive, downloaded)
    assert (downloaded / destination.name / "checkpoint.json").is_file()
    result = downloaded / destination.name / "result.json"
    marker = json.loads((result.parent / "task.finished.json").read_text())
    assert marker["result_checksum"] == file_checksum(result)


def test_batch_can_defer_per_task_archives(tmp_path):
    task = terminal_task(tmp_path / "batch/00000-MC-A", "completed")
    results = tmp_path / "results"
    export_batch_result(task, results, create_archive=False)
    assert not list(results.glob("*.zip"))


def test_single_worker_places_archive_in_registered_stage_results(tmp_path):
    from phase_agent.tools.remote.run_single_task import run_single_task

    stage = tmp_path / "MC-0001"
    batch = stage / "inputs" / "batch"
    task = batch / "00000-MC-A"
    task.mkdir(parents=True)
    (batch / "batch.snapshot.json").write_text("{}")
    payload = {
        "task_id": "A",
        "task_key": "key",
        "stage": "deep_search",
        "calculation_directory": ".",
    }
    (task / "task.json").write_text(json.dumps(payload))
    run_single_task(
        task / "task.json", executor=lambda t: {"status": "failed", "outputs": {}}
    )
    assert len(list((stage / "results").glob("*.results.zip"))) == 1
    assert not (stage / "inputs/results").exists()


def test_standalone_worker_also_compresses_terminal_result(tmp_path):
    from phase_agent.tools.remote.run_single_task import run_single_task

    task = tmp_path / "standalone"
    task.mkdir()
    (task / "task.json").write_text(
        json.dumps({"task_id": "A", "task_key": "key", "stage": "deep_search"})
    )
    run_single_task(
        task / "task.json", executor=lambda t: {"status": "completed", "outputs": {}}
    )
    assert (tmp_path / "standalone.results.zip").is_file()


def test_portable_jobs_carry_updated_exporter(tmp_path):
    from phase_agent.tools.remote.manual_upload_runner import _portable_transfer_worker

    _portable_transfer_worker(tmp_path, "run_mlip_batch", "_handoff_mlip_batch")
    batch = (tmp_path / "_handoff_mlip_batch.py").read_text(encoding="utf-8")
    assert "from _handoff_export_batch_result import export_batch_result" in batch
    exporter = (tmp_path / "_handoff_export_batch_result.py").read_text(
        encoding="utf-8"
    )
    assert "create_archive=True" in exporter
    assert "from transfer_bundle import result_bundle" in exporter


def test_completed_batch_automatically_emits_one_archive(tmp_path, monkeypatch):
    from phase_agent.tools.remote.run_mlip_batch import run_mlip_batch

    batch = tmp_path / "stage/inputs/batch"
    rows = []
    for index in range(2):
        task = terminal_task(batch / f"{index:05d}-MC-{index}", "completed")
        # Completed markers cause the batch runner to reuse the real terminal
        # files, without starting a calculation subprocess.
        rows.append({"task_id": "A", "input_path": f"{task.name}/task.json"})
    (batch / "manifest.json").write_text(json.dumps(rows))
    assert run_mlip_batch(batch / "manifest.json") == []
    archives = list((tmp_path / "stage/results").glob("*.results.zip"))
    assert [p.name for p in archives] == ["batch.results.zip"]

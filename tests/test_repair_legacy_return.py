import json
from pathlib import Path

import pytest

from phase_agent.runtime import repair_legacy_return as repair
from phase_agent.tools.remote.transfer_bundle import checksum, create_bundle


@pytest.mark.parametrize("bad", [None, "job", "input", "marker", "result_job"])
@pytest.mark.parametrize("completed", [False, True])
def test_repair_requires_submission_and_terminal_identity(tmp_path, monkeypatch, bad, completed):
    output = tmp_path / "Batch_output"
    output.mkdir()
    payload = tmp_path / "payload" / "results" / "T"
    payload.mkdir(parents=True)
    expected = {"task_id": "T", "task_key": "K", "task_checksum": "C"}
    data = {**expected, "status": "completed" if completed else "failed", "runtime_observation": {"job_id": "9" if bad == "result_job" else "7"}}
    if completed:
        training = payload / "training.json"
        training.write_text('{"frames":[]}', encoding="utf-8")
        data["outputs"] = {"training_file": "training.json", "training_checksum": checksum(training), "structure": {}}
    result = payload / "result.json"
    result.write_text(json.dumps(data), encoding="utf-8")
    marker = {**expected, "status": data["status"], "result_checksum": checksum(result)}
    if bad == "marker":
        marker["task_checksum"] = "wrong"
    (payload / "task.finished.json").write_text(json.dumps(marker), encoding="utf-8")
    archive = output / "legacy.results.zip"
    create_bundle(tmp_path / "payload", list(payload.iterdir()), archive, kind="results")
    receipt = tmp_path / "submitted.json"
    receipt.write_text(json.dumps({"archive_sha256": "wrong" if bad == "input" else "INPUT", "jobs": {"inputs/T/GPU.sh": {"job_id": "9" if bad == "job" else "7"}}}), encoding="utf-8")
    plan = {"source_folder": "Batch_input", "source_archive_sha256": "INPUT", "tasks": [{"expected": expected, "input_directory": "inputs/T", "result_directory": "results/T"}]}
    monkeypatch.setattr(repair, "recovery_plan", lambda state, h: plan)
    state = {"slurm_batches": [{"task_ids": ["T"], "handoff": {"return_directory": str(output)}}]}
    if bad:
        with pytest.raises(ValueError):
            repair.repair_failure_return(state, archive, receipt, task_id="T", job_id="7", include_completed=completed)
        assert not list(output.glob("verified-*.results.zip"))
    else:
        report = repair.repair_failure_return(state, archive, receipt, task_id="T", job_id="7", include_completed=completed)
        assert report["status"] == "verified"
        assert archive.is_file()
        assert report["science_submissions"] == 0


@pytest.mark.parametrize("missing_proof", [None, "feedback", "receipt", "changed_result"])
def test_acknowledge_source_requires_all_verified_and_imported_tasks(tmp_path, monkeypatch, missing_proof):
    from phase_agent.tools.remote.transfer_bundle import _archive_receipt, import_bundle

    output = tmp_path / "Batch_output"
    output.mkdir()
    payload = tmp_path / "payload" / "results" / "T"
    payload.mkdir(parents=True)
    expected = {"task_id": "T", "task_key": "K", "task_checksum": "C"}
    result = payload / "result.json"
    result.write_text(json.dumps({**expected, "status": "failed", "runtime_observation": {"job_id": "7"}}), encoding="utf8")
    (payload / "task.finished.json").write_text(json.dumps({**expected, "status": "failed", "result_checksum": checksum(result)}), encoding="utf8")
    archive = output / "legacy.results.zip"
    create_bundle(tmp_path / "payload", list(payload.iterdir()), archive, kind="results")
    submitted = output / "submitted.json"
    submitted.write_text(json.dumps({"archive_sha256": "INPUT", "jobs": {"inputs/T/GPU.sh": {"job_id": "7"}}}), encoding="utf8")
    plan = {"source_folder": "Batch_input", "source_archive_sha256": "INPUT", "tasks": [{"expected": expected, "input_directory": "inputs/T", "result_directory": "results/T"}]}
    monkeypatch.setattr(repair, "recovery_plan", lambda state, h: plan)
    destination = tmp_path / "destination"
    input_folder = tmp_path / "Batch_input"
    input_folder.mkdir()
    input_archive = input_folder / "inputs.zip"
    input_archive.write_bytes(b"registered input")
    handoff = {"layout_version": 4, "folder": str(input_folder), "upload_archive": str(input_archive),
               "return_directory": str(output), "import_directory": str(destination)}
    state = {"slurm_batches": [{"task_ids": ["T"], "handoff": handoff}], "processed_task_ids": ["T"], "feedback_processed_task_ids": ["T"]}
    report = repair.repair_failure_return(state, archive, submitted, task_id="T", job_id="7", include_completed=True)
    repaired = Path(report["repaired_archive"])
    import_bundle(repaired, destination)
    receipt = _archive_receipt(output, repaired)
    receipt.parent.mkdir()
    receipt.write_text(checksum(repaired), encoding="utf8")
    subset = output / "subset.results.zip"
    create_bundle(tmp_path / "payload", [result], subset, kind="results")
    if missing_proof == "feedback":
        state["feedback_processed_task_ids"] = []
    elif missing_proof == "receipt":
        receipt.unlink()
    elif missing_proof == "changed_result":
        (destination / "results/T/result.json").write_text("changed", encoding="utf8")
    from phase_agent.tools.remote.batch_runner import RemoteBatchRunner
    result = RemoteBatchRunner(tmp_path / "unused", worker_command=[]).collect_results_with_report(state)
    remaining = result["report"]["transfer_errors"]
    if missing_proof:
        source_error = next(row for row in remaining if Path(row["archive"]) == archive)
        assert source_error.get("legacy_repair_blocked")
        assert not _archive_receipt(output, archive).exists()
        assert not _archive_receipt(output, subset).exists()
    else:
        assert remaining == []
        assert _archive_receipt(output, archive).read_text(encoding="utf8") == checksum(archive)
        assert _archive_receipt(output, subset).read_text(encoding="utf8") == checksum(subset)


@pytest.mark.parametrize("bad", [None, "unreceipted", "different"])
def test_old_subset_requires_receipted_identical_source(tmp_path, bad):
    from phase_agent.tools.remote.transfer_bundle import _archive_receipt, import_bundle

    output = tmp_path / "Batch_output"
    output.mkdir()
    payload = tmp_path / "payload"
    payload.mkdir()
    item = payload / "result.json"
    item.write_text('{"result":1}', encoding="utf8")
    reference = output / "full.results.zip"
    create_bundle(payload, [item], reference, kind="results")
    destination = tmp_path / "destination"
    import_bundle(reference, destination)
    receipt = _archive_receipt(output, reference)
    receipt.parent.mkdir()
    receipt.write_text(checksum(reference), encoding="utf8")
    if bad == "unreceipted":
        receipt.unlink()
    if bad == "different":
        item.write_text('{"result":2}', encoding="utf8")
    subset = output / "subset.results.zip"
    create_bundle(payload, [item], subset, kind="results")
    state = {"slurm_batches": [{"handoff": {"return_directory": str(output), "import_directory": str(destination)}}]}
    if bad:
        with pytest.raises(ValueError):
            repair.acknowledge_verified_subset(state, subset, reference)
        assert not _archive_receipt(output, subset).exists()
    else:
        assert repair.acknowledge_verified_subset(state, subset, reference)["files"] == 1
        assert _archive_receipt(output, subset).read_text(encoding="utf8") == checksum(subset)

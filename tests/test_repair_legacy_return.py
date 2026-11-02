import json

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

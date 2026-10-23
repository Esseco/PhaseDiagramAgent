from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from phase_agent.tools.relax_mc_batch import dependencies
from phase_agent.tools.relax_mc_batch.files import read
from phase_agent.tools.remote.manual_upload_runner import ManualUploadBatchRunner
from tests.test_relax_mc_phase_dependencies import setup, run_batch


@pytest.mark.parametrize("error_key", ["result_validation_errors", "scientific_feedback_errors"])
def test_phase_requires_feedback_acceptance_before_marking_done(tmp_path, monkeypatch, error_key):
    state = {
        "tasks": [{"task_id": "t", "status": "pending"}],
        "slurm_batches": [{"batch_id": "b", "job_id": "123", "task_ids": ["t"]}],
        "relax_mc_remote": {"phase": 1, "phase_controllers": {"1": {"batch_ids": ["b"]}}},
    }
    monkeypatch.setattr(
        ManualUploadBatchRunner,
        "collect_results",
        lambda *a: [{"task_id": "t", "status": "completed"}],
    )

    def reject(current, *a):
        current = deepcopy(current)
        if error_key == "scientific_feedback_errors":
            current["tasks"][0]["status"] = "completed"
        current[error_key] = {"t": {"code": "result_output_object_required:pool_summary"}}
        return current

    monkeypatch.setattr(dependencies, "feedback", reject)
    io = SimpleNamespace(active=lambda ids: set())
    current, results = dependencies.collect_phase(
        tmp_path,
        state,
        None,
        {
            "config": {"supercomputer": {"worker": {"command": []}}},
            "phase_references": {},
            "model_version": "m",
        },
        io,
    )
    assert results[0]["status"] == "completed"
    assert not current["slurm_batches"][0].get("relax_mc_wave_done")
    diagnostic = current["relax_mc_remote"]["failure_diagnostics"]
    assert diagnostic["unverified_task_ids"] == ["t"]
    assert diagnostic["issues"][0]["origin"] == "remote_validation"
    assert diagnostic["issues"][0]["code"] == "result_output_object_required:pool_summary"
    assert diagnostic["issues"][0]["job_id"] == "123"


@pytest.mark.parametrize("batch_ids", [[], ["missing"]])
def test_empty_or_missing_phase_scope_is_not_success(tmp_path, monkeypatch, batch_ids):
    state = {
        "tasks": [{"task_id": "t", "status": "pending"}],
        "slurm_batches": [],
        "relax_mc_remote": {"phase": 1, "phase_controllers": {"1": {"batch_ids": batch_ids}}},
    }
    monkeypatch.setattr(ManualUploadBatchRunner, "collect_results", lambda *a: [])
    with pytest.raises(ValueError, match="phase_task_scope_invalid"):
        dependencies.collect_phase(
            tmp_path,
            state,
            None,
            {"config": {"supercomputer": {"worker": {"command": []}}}},
            SimpleNamespace(active=lambda ids: set()),
        )


@pytest.mark.parametrize(
    "fault",
    ["pending", "running", "result_validation_errors", "scientific_feedback_errors", "failed"],
)
def test_finish_never_reports_incomplete_batch_as_completed(tmp_path, monkeypatch, fault):
    state = {"tasks": [{"task_id": "t", "status": "completed"}], "relax_mc_remote": {}}
    if fault.endswith("errors"):
        state[fault] = {"t": {"code": "rejected"}}
    else:
        state["tasks"][0]["status"] = fault
    monkeypatch.setattr(dependencies, "checkpoint", lambda *a: None)
    monkeypatch.setattr(dependencies, "publish", lambda *a: "results.zip")
    monkeypatch.setattr(
        "phase_agent.tools.relax_mc_batch.upload_driver.publish_output", lambda *a: None
    )
    result = dependencies.finish(tmp_path, state, None, {}, "completed")
    assert result["status"] == ("partial" if fault == "failed" else "paused")
    assert state["relax_mc_remote"]["status"] == result["status"]


def test_tasks_in_progress_without_actions_is_not_completion(tmp_path, monkeypatch):
    root, io, _ = setup(tmp_path, monkeypatch)
    run_batch(root, submit=True, io=io)
    count = len(io.rows)
    monkeypatch.setattr(
        dependencies,
        "next_mc",
        lambda state, *a: {
            "status": "tasks_in_progress",
            "actions": [],
            "state": state,
        },
    )
    result = run_batch(root, submit=True, io=io, phase=0)
    assert result["status"] == "paused"
    assert result["reason"] == "tasks_in_progress"
    assert len(io.rows) == count


@pytest.mark.parametrize("driver", ["dependencies", "synchronous"])
def test_rejected_mc_stops_before_next_wave_and_returns_diagnostic(tmp_path, monkeypatch, driver):
    from phase_agent.persistence.ledger import result_validation
    from phase_agent.tools.relax_mc_batch import run
    from phase_agent.tools.relax_mc_batch.results import inspect_archive
    from tests.test_relax_mc_batch import package, fake_executor

    validate = result_validation.validate_scientific_result

    def old_validator(result):
        if result.get("stage") == "deep_search" and result.get("status") == "completed":
            raise result_validation.InvalidScientificResult(
                "result_output_object_required:pool_summary"
            )
        return validate(result)

    monkeypatch.setattr(result_validation, "validate_scientific_result", old_validator)
    if driver == "dependencies":
        root, io, calls = setup(tmp_path, monkeypatch)
        run_batch(root, submit=True, io=io)
        assert run_batch(root, submit=True, io=io, phase=0)["status"] == "queued"
        submissions = len(io.rows)
        outcome = run_batch(root, submit=True, io=io, phase=1)
        assert len(io.rows) == submissions
    else:
        _, _, _, _, _, prepared = package(tmp_path)
        execute, calls = fake_executor(monkeypatch)
        root = Path(prepared["directory"])
        outcome = run.run_batch(root, submit=True, batch_executor=execute)
    assert outcome["status"] == "paused"
    current = read(root / "workflow_state.json")
    assert current["relax_mc_remote"]["phase"] == 1
    assert current["relax_mc_remote"]["completed_mc_waves"] == []
    assert all(segment == 0 for stage, segment, steps in calls if stage == "deep_search")
    assert not any(
        b.get("relax_mc_wave_done")
        for b in current["slurm_batches"]
        if b.get("calculation_group") == "MC-search"
    )
    summary = inspect_archive(outcome["archive"])["summary"]
    groups = summary["diagnostics"]["issue_groups"]
    assert any(g["code"] == "result_output_object_required:pool_summary" for g in groups)


@pytest.mark.parametrize("status", ["completed", "partial", "paused", "failed", None])
def test_imported_batch_with_pending_mc_always_requests_collection_not_submission(tmp_path, status):
    from phase_agent.runtime.waiting_files_presentation import waiting_file_lines

    directory = tmp_path / "batch"
    folder = tmp_path / "batch_input"
    folder.mkdir()
    for name in ("submit_all.sh", "inputs.zip", "run_upload.py"):
        (folder / name).write_text("present")
    state = {
        "tasks": [
            {
                "task_id": "mc",
                "stage": "deep_search",
                "status": "pending",
                "relax_mc_bundle_id": "b",
            }
        ],
        "relax_mc_batches": {
            "b": {
                "directory": str(directory),
                "import_checksum": "verified",
                "result_summary": {"status": status},
            }
        },
    }
    before = deepcopy(state)
    text = "\n".join(waiting_file_lines(state))
    assert "仍缺 1 项结果的核验凭据" in text
    assert "不要重新上传输入或提交整批" in text
    assert "sbatch submit_all.sh" not in text
    assert state == before


@pytest.mark.parametrize("summary", [[], [{"rank": 0}]])
def test_new_upload_contains_validator_for_worker_pool_list(tmp_path, summary):
    import zipfile
    from tests.test_relax_mc_batch import package

    *_, prepared = package(tmp_path)
    with zipfile.ZipFile(prepared["upload_archive"]) as archive:
        source = archive.read("runtime/phase_agent/persistence/ledger/result_validation.py")
    namespace = {"__name__": "frozen_result_validation"}
    exec(compile(source, "frozen_result_validation.py", "exec"), namespace)
    namespace["validate_scientific_result"]({"outputs": {"pool_summary": summary}})


@pytest.mark.parametrize("summary", [[], {}])
def test_collect_only_checks_scientific_contract_before_claiming_recovered(tmp_path, summary):
    import hashlib
    import json
    from tests.test_collect_only_recovery import fixture, build
    from phase_agent.persistence.ledger.result_validation import InvalidScientificResult

    archive, evidence, job = fixture(tmp_path)
    path = evidence / "jobs/results/00000-t1/result.json"
    data = json.loads(path.read_text())
    data["outputs"].update(energy_unit="eV", pool_summary=summary)
    path.write_text(json.dumps(data))
    marker = path.with_name("task.finished.json")
    receipt = json.loads(marker.read_text())
    receipt["result_checksum"] = hashlib.sha256(path.read_bytes()).hexdigest()
    marker.write_text(json.dumps(receipt))
    if isinstance(summary, dict):
        with pytest.raises(InvalidScientificResult, match="pool_summary"):
            build(tmp_path, archive, evidence, job)
    else:
        assert build(tmp_path, archive, evidence, job)["recovery"]["recovered_task_ids"] == ["t1"]


@pytest.mark.parametrize("phase", [2, 3])
def test_collect_only_uses_verified_segments_when_old_controller_advanced_phase(tmp_path, phase):
    import hashlib
    import json
    from tests.test_collect_only_recovery import fixture, build
    from phase_agent.tools.remote.transfer_bundle import create_bundle

    _, evidence, job = fixture(tmp_path)
    original = tmp_path / "original"
    path = original / "workflow_state.json"
    data = json.loads(path.read_text())
    data["tasks"][0]["segment_index"] = 0
    path.write_text(json.dumps(data))
    archive = create_bundle(
        original, list(original.glob("*.json")), tmp_path / "segmented.zip", kind="results"
    )
    snapshot = evidence / "workflow_state.json"
    data = json.loads(snapshot.read_text())
    data["relax_mc_remote"]["phase"] = phase
    snapshot.write_text(json.dumps(data))
    result = evidence / "jobs/results/00000-t1/result.json"
    data = json.loads(result.read_text())
    data["outputs"]["energy_unit"] = "eV"
    result.write_text(json.dumps(data))
    marker = result.with_name("task.finished.json")
    receipt = json.loads(marker.read_text())
    receipt["result_checksum"] = hashlib.sha256(result.read_bytes()).hexdigest()
    marker.write_text(json.dumps(receipt))
    outcome = build(tmp_path, archive, evidence, job)
    assert outcome["status"] == "partial"
    assert outcome["recovery"]["next_phase_not_submitted"]
    assert read(Path(outcome["archive"]).parent / "result_summary.json")["completed_mc_waves"] == [
        0
    ]


def test_failed_mc_is_not_recorded_as_completed_wave(tmp_path, monkeypatch):
    from tests.test_relax_mc_batch import package, fake_executor
    from phase_agent.tools.relax_mc_batch.run import run_batch as run_synchronous

    *_, prepared = package(tmp_path)
    execute, calls = fake_executor(monkeypatch, fail_mc=True)
    outcome = run_synchronous(prepared["directory"], submit=True, batch_executor=execute)
    assert outcome["status"] == "paused"
    state = read(Path(prepared["directory"]) / "workflow_state.json")
    assert state["relax_mc_remote"]["completed_mc_waves"] == []
    assert state["relax_mc_remote"]["phase"] == 1

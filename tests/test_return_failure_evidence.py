from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import zipfile

import pytest

from phase_agent.tools.relax_mc_batch.failure_diagnostics import (
    compact_diagnostics,
    read_return_diagnostics,
    record_remote_diagnostics,
)
from phase_agent.tools.relax_mc_batch.files import write
from phase_agent.tools.relax_mc_batch.results import import_results, recover_available
from phase_agent.tools.relax_mc_batch.run import SlurmIO
from phase_agent.tools.remote.transfer_bundle import create_bundle, checksum


def state_with_task(tmp_path):
    script = tmp_path / "gpu.sh"
    script.write_text("#SBATCH --output=slurm-%j.out\n")
    return {
        "tasks": [{"task_id": "t", "status": "pending"}],
        "slurm_batches": [
            {"batch_id": "b", "job_id": "123", "task_ids": ["t"], "script_path": str(script)}
        ],
        "relax_mc_remote": {"phase": 0, "phase_controllers": {"0": {"batch_ids": ["b"]}}},
    }


def test_worker_failure_retains_original_exception_and_bounded_traceback(tmp_path, monkeypatch):
    from phase_agent.tools.remote.run_single_task import run_single_task

    monkeypatch.setenv("PHASE_DEFER_RESULT_ARCHIVE", "1")
    task = tmp_path / "task.json"
    write(task, {"task_id": "t", "task_key": "k"})

    def fail(task):
        raise FileNotFoundError("/cluster/finetune/com_1/com_1.model")

    result = run_single_task(task, executor=fail)
    report = record_remote_diagnostics(tmp_path, state_with_task(tmp_path), [result], [], {}, None)
    issue = compact_diagnostics(report, "return_diagnostics.json")["issue_groups"][0]
    assert issue["origin"] == "remote_calculation"
    assert "com_1.model" in issue["error"] and "FileNotFoundError" in issue["traceback"]
    assert issue["task_ids"] == ["t"]


@pytest.mark.parametrize(
    "job_state,origin",
    [
        ("OUT_OF_MEMORY", "remote_job"),
        ("COMPLETED", "remote_collection"),
        ("UNKNOWN", "remote_collection"),
    ],
)
def test_missing_receipt_keeps_task_pending_and_preserves_collection_cause(
    tmp_path, monkeypatch, job_state, origin
):
    from phase_agent.tools.relax_mc_batch import dependencies
    from phase_agent.tools.remote.manual_upload_runner import ManualUploadBatchRunner

    state = state_with_task(tmp_path)
    (tmp_path / "slurm-123.out").write_text("x" * 20000 + "\nCUDA allocation failed\n")

    def collect(self, current):
        self.last_collection_report = {
            "details_preview": [{"task_id": "t", "reason": "result_checksum_mismatch"}]
        }
        return []

    monkeypatch.setattr(ManualUploadBatchRunner, "collect_results", collect)
    feedback = Mock(side_effect=AssertionError("no unverified task settlement"))
    monkeypatch.setattr(dependencies, "feedback", feedback)
    io = SimpleNamespace(
        active=lambda ids: set(),
        evidence=lambda ids: {"123": {"state": job_state, "exit_code": "0:9"}},
    )
    current, rows = dependencies.collect_phase(
        tmp_path, state, None, {"config": {"supercomputer": {"worker": {"command": []}}}}, io
    )
    assert not rows and current["tasks"][0]["status"] == "pending"
    diagnostic = current["relax_mc_remote"]["failure_diagnostics"]
    assert diagnostic["issues"][0]["origin"] == origin
    assert diagnostic["issues"][0]["code"] == "result_checksum_mismatch"
    assert "CUDA allocation failed" in diagnostic["jobs"][0]["logs"][0]["log_tail"]
    assert len(diagnostic["jobs"][0]["logs"][0]["log_tail"]) <= 8192
    feedback.assert_not_called()
    assert not current["slurm_batches"][0].get("relax_mc_wave_done")


def test_slurm_exit_code_is_returned_without_interpreting_it_as_task_completion(monkeypatch):
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **k: SimpleNamespace(
            returncode=0, stdout="123|OUT_OF_MEMORY|0:9|MemoryLimit\n", stderr=""
        ),
    )
    assert SlurmIO().evidence(["123"])["123"] == {
        "state": "OUT_OF_MEMORY",
        "exit_code": "0:9",
        "reason": "MemoryLimit",
    }


def returned_archive(tmp_path):
    root = tmp_path / "remote"
    root.mkdir()
    summary = {
        "bundle_id": "b",
        "plan_checksum": "plan",
        "model_version": "m1",
        "config_version": "c1",
        "status": "failed",
    }
    write(root / "result_summary.json", summary)
    write(root / "workflow_state.json", {"tasks": [{"task_id": "t", "status": "pending"}]})
    write(root / "failure.json", {"reason": "squeue query failed", "traceback": "controller stack"})
    archive = tmp_path / "results.zip"
    create_bundle(root, list(root.glob("*.json")), archive, kind="results")
    destination = tmp_path / "verified"
    record = {
        **summary,
        "branch_ids": [],
        "expected_results_archive": str(archive),
        "return_directory": str(destination),
        "result_summary": summary,
        "import_checksum": checksum(archive),
    }
    state = {"active_model_version": "m1", "relax_mc_batches": {"b": record}}
    context = {
        "event_state": state,
        "config_version": "c1",
        "effective_config": {"mlip": {"version": "m1"}},
    }
    return archive, destination, state, context


def test_same_archive_repairs_missing_local_extraction_without_compute_or_readmission(
    tmp_path, monkeypatch
):
    archive, destination, state, context = returned_archive(tmp_path)
    monkeypatch.setattr(
        "phase_agent.tools.relax_mc_batch.results.reconcile_task_results",
        Mock(side_effect=AssertionError("must not readmit")),
    )
    result = import_results(action={"parameters": {"bundle_id": "b"}}, context=context)
    assert result["status"] == "already_imported" and not result["submitted"]
    assert (destination / "failure.json").is_file()
    record = result["state"]["relax_mc_batches"]["b"]
    assert record["local_recovery"]["restored_file_count"] == 3
    assert record["remote_diagnostics"]["controller"]["error"] == "squeue query failed"
    assert record["remote_diagnostics"]["unverified_task_ids"] == ["t"]
    assert state == context["event_state"] and not state["relax_mc_batches"]["b"].get(
        "local_recovery"
    )
    assert (
        import_results(
            action={"parameters": {"bundle_id": "b"}},
            context={**context, "event_state": result["state"]},
        )["status"]
        == "already_imported"
    )


def test_corrupt_local_files_are_not_overwritten_when_repairing_missing_files(tmp_path):
    archive, destination, state, context = returned_archive(tmp_path)
    destination.mkdir()
    (destination / "failure.json").write_text("keep this conflicting evidence")
    result = recover_available(state, context)
    diagnostic = result["relax_mc_batches"]["b"]["return_diagnostic"]
    assert diagnostic["origin"] == "local_import"
    assert "Existing result differs" in diagnostic["error"]
    assert (destination / "failure.json").read_text() == "keep this conflicting evidence"


def test_remote_driver_failure_is_not_called_a_local_import_failure(tmp_path):
    folder = tmp_path / "batch"
    output = tmp_path / "batch_output"
    output.mkdir()
    with zipfile.ZipFile(output / "diagnostics.zip", "w") as z:
        z.writestr(
            "run_status.json", json.dumps({"returncode": 1, "error": "model preflight failed"})
        )
        z.writestr("run.log", "original remote stack")
    state = {
        "relax_mc_batches": {
            "b": {"directory": str(folder), "expected_results_archive": str(output / "results.zip")}
        }
    }
    current = recover_available(state, {})
    diagnostic = current["relax_mc_batches"]["b"]["return_diagnostic"]
    assert diagnostic["origin"] == "remote_controller" and diagnostic["exit_code"] == 1
    assert diagnostic["log_tail"] == "original remote stack"


def test_compact_remote_diagnostics_are_present_in_agent_batch_context(tmp_path):
    from phase_agent.tools.relax_mc_batch.plan import batch_facts
    from phase_agent.runtime.result_recovery_status import result_recovery_facts

    diagnostic = {"counts": {"remote_calculation": 1}, "issue_groups": [{"error": "missing model"}]}
    state = {"relax_mc_batches": {"b": {"remote_diagnostics": diagnostic}}}
    assert batch_facts(state)[0]["remote_diagnostics"] == diagnostic
    assert result_recovery_facts(state)[0]["remote_diagnostics"] == diagnostic


def test_legacy_synthetic_failure_without_receipt_is_not_confirmed_computation_error(tmp_path):
    task = {"task_id": "t", "status": "failed", "error": "ended_without_verified_result"}
    diagnostic = read_return_diagnostics(tmp_path, {}, [task])
    assert diagnostic["issue_groups"][0]["origin"] == "remote_collection"
    assert diagnostic["issue_groups"][0]["code"] == "legacy_failure_unverified"
    from phase_agent.runtime.result_recovery_status import result_recovery_facts

    task["relax_mc_bundle_id"] = "b"
    state = {
        "tasks": [task],
        "relax_mc_batches": {
            "b": {
                "remote_diagnostics": diagnostic,
                "import_checksum": "checked",
                "result_summary": {"failed_task_ids": ["t"]},
            }
        },
    }
    facts = result_recovery_facts(state)[0]
    assert not facts["verified_failed_task_ids"]
    assert facts["unverified_failure_task_ids"] == ["t"]
    assert facts["kind"] == "needs_review"


@pytest.mark.parametrize("fault", ["binding", "issues", "jobs", "text", "scope"])
def test_diagnostic_payload_is_validated_before_agent_use(tmp_path, fault):
    summary = {"bundle_id": "b", "plan_checksum": "p", "model_version": "m", "config_version": "c"}
    report = {
        **summary,
        "schema_version": 1,
        "issues": [{"task_id": "t", "origin": "remote_calculation", "error": "failed"}],
    }
    if fault == "binding":
        report["model_version"] = "wrong"
    elif fault in {"issues", "jobs"}:
        report[fault] = ["wrong"]
    elif fault == "text":
        report["issues"][0]["error"] = {"wrong": "shape"}
    else:
        report["issues"][0]["task_id"] = "outside"
    write(tmp_path / "return_diagnostics.json", report)
    with pytest.raises(ValueError, match="diagnostic"):
        read_return_diagnostics(tmp_path, summary, [{"task_id": "t"}])


def test_remote_publication_includes_bound_machine_readable_diagnostics(tmp_path):
    from phase_agent.tools.relax_mc_batch.run import publish
    from phase_agent.tools.relax_mc_batch.results import inspect_archive

    state = state_with_task(tmp_path)
    state["active_model_version"] = "m"
    record_remote_diagnostics(tmp_path, state, [], ["t"], {}, None)
    write(tmp_path / "workflow_state.json", {"tasks": state["tasks"]})
    manifest = {"bundle_id": "b", "plan_checksum": "p", "model_version": "m", "config_version": "c"}
    archive = publish(tmp_path, state, manifest, "paused", "remote_results_unverified")
    with zipfile.ZipFile(archive) as z:
        report = json.loads(z.read("return_diagnostics.json"))
        assert report["unverified_task_ids"] == ["t"]
        assert report["bundle_id"] == "b" and report["model_version"] == "m"
        assert report["issues"][0]["origin"] == "remote_collection"
    assert inspect_archive(archive)["summary"]["diagnostics"]["unverified_task_ids"] == ["t"]


def test_latest_driver_error_is_read_even_when_old_result_archive_is_already_imported(tmp_path):
    archive, destination, state, context = returned_archive(tmp_path)
    with zipfile.ZipFile(tmp_path / "diagnostics.zip", "w") as z:
        z.writestr(
            "run_status.json", json.dumps({"returncode": 1, "error": "new remote export failure"})
        )
        z.writestr("run.log", "export stack from compute node")
    current = recover_available(state, context)
    diagnostic = current["relax_mc_batches"]["b"]["driver_diagnostic"]
    assert diagnostic["origin"] == "remote_controller"
    assert diagnostic["error"] == "new remote export failure"
    assert diagnostic["log_tail"] == "export stack from compute node"
    assert current["relax_mc_batches"]["b"]["import_checksum"] == checksum(archive)

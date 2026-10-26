"""Regression checks for numerical and file boundaries found in the repo review."""

import json
from pathlib import Path
import zipfile

from ase import Atoms
import numpy as np
import pytest

from phase_agent.analysis.convergence.check_global_convergence import check_global_convergence
from phase_agent.science.qbc.generate_qbc_file import generate_qbc_file
from phase_agent.tools.dispatch.execute_tool_action import execute_tool_action
from phase_agent.tools.remote.integrity import file_checksum
from phase_agent.tools.remote.submission_storage import is_storage_archive
from phase_agent.tools.remote.transfer_bundle import (
    MANIFEST,
    create_bundle,
    has_registered_return,
    import_bundle,
    import_registered_bundles,
)
from phase_agent.tools.slurm.run_slurm_array_task import run_slurm_array_task
from phase_agent.tools.slurm.slurm_batch_runner import SlurmBatchRunner


@pytest.mark.parametrize("accepted", [False, True])
def test_dispatched_convergence_preserves_acceptance(accepted):
    state = {
        "tasks": [],
        "user_accepted_convergence": accepted,
        "model_update_epochs": [
            {"model_version": "m1", "hull_change": 0.001, "ground_state_unchanged": True}
        ],
        "final_frame_dft_errors": [{"error_ev_per_atom": 0.001}],
    }
    raw = check_global_convergence(
        state,
        rules={
            "hull_change_tolerance": 0.003,
            "stable_model_update_epochs": 1,
            "final_energy_mae_tolerance": 0.003,
            "minimum_recent_dft_checks": 1,
        },
    )
    executed = execute_tool_action(
        {"tool": "check_convergence", "task_key": "c1"},
        registry={"check_convergence": {"handler": lambda **_: raw}},
        context={},
    )
    assert executed["status"] == "completed"
    result = executed["result"]
    assert result["criteria_satisfied"] is True
    assert result["converged"] is accepted
    assert result["requires_user_acceptance"] is (not accepted)
    assert result["search_status"] == ("converged" if accepted else "numerical_criteria_satisfied")


@pytest.mark.parametrize("field", ["energy", "forces"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_qbc_rejects_nonfinite_predictions(tmp_path, field, value):
    def predictor(**kwargs):
        result = {"energy": 0.0, "forces": np.zeros((1, 3))}
        if kwargs["model"] == 1:
            result[field] = value if field == "energy" else [[value, 0.0, 0.0]]
        return result

    report = generate_qbc_file(
        [{"structure_id": "S1", "structure": Atoms("H", positions=[[0, 0, 0]])}],
        {"loaded_members": [{"model_id": str(i), "model": i} for i in range(3)]},
        tmp_path / "qbc.json",
        predictor=predictor,
    )
    assert report["status"] == "failed"
    assert report["results"][0]["qbc"]["status"] == "failed"
    assert "非有限值" in report["results"][0]["qbc"]["error"]
    json.dumps(report, allow_nan=False)


def _slurm(tmp_path):
    tasks = [
        {
            "task_id": "T1",
            "task_key": "K1",
            "stage": "simple_check",
            "status": "pending",
            "config_version": "c1",
            "model_version": "m1",
        }
    ]
    runner = SlurmBatchRunner(tmp_path, worker_command=["fake-worker"])
    prepared = runner.prepare(
        {
            "tasks": tasks,
            "budget_reservations": {"K1": {"status": "reserved", "reserved_cost": 2.0}},
        }
    )
    run_slurm_array_task(
        prepared["batch"]["manifest_path"],
        0,
        executor=lambda _: {"status": "completed", "actual_cost": 1.0},
    )
    return runner, prepared["state"], Path(prepared["state"]["tasks"][0]["result_path"])


def test_slurm_accepts_valid_protocol_with_versions(tmp_path):
    runner, state, _ = _slurm(tmp_path)
    (result,) = runner.collect_results(state)
    assert result["task_id"] == "T1" and result["config_version"] == "c1"
    assert result["model_version"] == "m1" and result["actual_cost"] == 1.0


@pytest.mark.parametrize("mutation", ["checksum", "task_id", "marker", "invalid_json"])
def test_slurm_rejects_invalid_results(tmp_path, mutation):
    runner, state, path = _slurm(tmp_path)
    marker_path = path.with_name("task.finished.json")
    payload = json.loads(path.read_text())
    marker = json.loads(marker_path.read_text())
    if mutation == "checksum":
        payload["actual_cost"] = 9.0
    elif mutation == "task_id":
        payload["task_id"] = marker["task_id"] = "UNREGISTERED"
    elif mutation == "marker":
        marker = []
    if mutation == "invalid_json":
        path.write_text("{", encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")
    if mutation == "task_id":
        marker["result_checksum"] = file_checksum(path)
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    assert runner.collect_results(state) == []
    assert runner.last_collection_report["invalid_count"] == 1
    assert runner.last_collection_report["details_preview"][0]["task_id"] == "T1"
    assert state["budget_reservations"]["K1"]["status"] == "reserved"


@pytest.mark.parametrize(
    "manifest",
    [
        [],
        None,
        "text",
        {"schema": 1, "kind": [], "files": []},
        {"schema": 1, "kind": "results", "files": [None]},
    ],
)
def test_bad_manifest_is_reported_without_blocking_other_returns(tmp_path, manifest):
    root = tmp_path / "returns"
    root.mkdir()
    with zipfile.ZipFile(root / "a-bad.results.zip", "w") as archive:
        archive.writestr(MANIFEST, json.dumps(manifest))
    source = tmp_path / "result.json"
    source.write_text("{}")
    create_bundle(tmp_path, [source], root / "b-good.results.zip", kind="results")
    assert has_registered_return({"slurm_batches": [{"results_directory": str(root)}]})
    errors = import_registered_bundles([root])
    assert len(errors) == 1 and "a-bad.results.zip" in errors[0]["archive"]
    assert (root / "result.json").is_file()
    assert (root / ".transfer_receipts/b-good.results.zip.sha256").is_file()
    with pytest.raises(ValueError):
        import_bundle(root / "a-bad.results.zip", tmp_path / "isolated")


def test_manifest_discovery_checks_size_before_read(tmp_path, monkeypatch):
    import phase_agent.tools.remote.transfer_bundle as transfer

    archive_path = tmp_path / "large.results.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(MANIFEST, json.dumps({"kind": "storage_results"}))
    monkeypatch.setattr(transfer, "MAX_MANIFEST_BYTES", 1)
    assert is_storage_archive(archive_path) is False


def test_confirmed_project_with_missing_environment_is_not_ready(tmp_path):
    from tests.test_runtime_doctor import project, write, files
    from phase_agent.runtime.project_diagnostics import inspect_runtime

    runtime = project(tmp_path)
    session_path = tmp_path / "parameters/config_session.json"
    session = json.loads(session_path.read_text(encoding="utf-8-sig"))
    config = session["config"]
    config.pop("storage", None)
    config["mlip"]["model_path"] = "/remote/model.model"
    config["python_environments"] = {
        "local_python": "py1",
        "local_mlip": "mace",
        "remote_python": "py1",
        "remote_mlip": "mace",
    }
    session.update(
        status="confirmed", confirmed_snapshot={"config": config, "config_version": "v1"}
    )
    reference = tmp_path / "structures/O3.vasp"
    reference.parent.mkdir()
    reference.write_text("reference")
    write(tmp_path / "workflow_state/state.json", {"tasks": [], "confirmed_config_version": "v1"})
    write(tmp_path / "workflow_state/ledgers/phase_data.json", {"branches": {}, "structures": {}})
    write(session_path, session)
    assert inspect_runtime(runtime)["status"] == "ready"
    config["python_environments"].pop("remote_mlip")
    write(session_path, session)
    before = files(tmp_path)
    result = inspect_runtime(runtime)
    assert result["sections"]["python_environments"]["data"]["missing"] == ["remote_mlip"]
    assert result["status"] == "needs_attention"
    assert files(tmp_path) == before


def test_compute_wave_uses_materialized_tasks_without_upload_archives(tmp_path, monkeypatch):
    from tests.test_relax_mc_batch import package, fake_executor
    from phase_agent.tools.relax_mc_batch.run import run_batch

    *_, prepared = package(tmp_path, count=1, single_x=True)
    execute, calls = fake_executor(monkeypatch)
    result = run_batch(prepared["directory"], submit=True, batch_executor=execute)
    assert result["status"] == "paused"  # Only one composition: MC needs hull endpoints.
    assert len(calls) == 1
    jobs = Path(prepared["directory"]) / "jobs"
    assert list(jobs.rglob("SHA256SUMS"))
    assert list(jobs.rglob("task.finished.json"))
    assert not list(jobs.rglob("*.upload.zip"))


def test_dft_product_publisher_handles_long_paths_and_keeps_unchanged_files(tmp_path):
    from phase_agent.analysis.feedback.export_dft_products import _publish_bytes
    from phase_agent.tools.remote.transfer_bundle import local_path

    path = tmp_path
    while len(str(path)) <= 280:
        path = path / ("nested_" * 8)
    path = path / "product.json"
    assert len(str(path)) > 260
    _publish_bytes(path, b'{"valid":true}')
    target = local_path(path)
    assert target.read_bytes() == b'{"valid":true}'
    timestamp = target.stat().st_mtime_ns
    _publish_bytes(path, b'{"valid":true}')
    assert target.stat().st_mtime_ns == timestamp
    assert not list(target.parent.glob("*.tmp"))

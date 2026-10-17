import gzip
import json
from pathlib import Path
import zipfile

import pytest
from phase_agent.tools.remote import legacy_recovery_runner as runner
from phase_agent.tools.remote.legacy_recovery import prepare_recovery_handoff, recovery_plan
from phase_agent.tools.remote.integrity import file_checksum
from tests.test_legacy_delivery_recovery import setup_original, remote_copy, fake_result, CPU


@pytest.fixture
def tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp("d")


def seed_logs(source, task):
    worker = source / task["input_directory"]
    worker.mkdir(parents=True, exist_ok=True)
    (worker / "workflow.json").write_text(json.dumps({"calculation": "static"}))
    (worker / "workflow_state.json").write_text(
        json.dumps(
            {
                "stages": {
                    "static": {
                        "directory": "runs/static/attempt_001",
                        "status": "failed",
                        "error": "Static output needs WAVECAR for subsequent calculations",
                    }
                }
            }
        )
    )
    (worker / "failure_report.txt").write_text(
        "Static output needs WAVECAR for subsequent calculations"
    )
    (worker / "workflow_status.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "exception": "ValueError: Static output needs WAVECAR for subsequent calculations",
            }
        )
    )
    calculation = worker / "runs/static/attempt_001"
    calculation.mkdir(parents=True)
    with gzip.open(calculation / "INCAR.gz", "wt") as stream:
        stream.write("LWAVE = False\nNSW = 0\n")
    (calculation / "std_err.txt").write_text("FORTRAN STOP")
    return worker, calculation


def test_diagnostics_collect_evidence_without_wavefunction_contents(tmp_path, monkeypatch):
    task = {"input_directory": "worker"}
    worker, calculation = seed_logs(tmp_path, task)
    (calculation / "CHGCAR").write_bytes(b"charge")
    (worker / "out").write_bytes(b"x" * 100000)
    original = Path.open

    def read_file(path, *args, **kwargs):
        if path.name in {"WAVECAR", "CHGCAR"}:
            pytest.fail("must never read restart file payload")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", read_file)
    result = runner.failure_diagnostics(tmp_path, task)
    assert "WAVECAR" in result["files"]["workflow_status.json"]["text"]
    assert "LWAVE = False" in result["files"]["calculation/INCAR"]["text"]
    assert result["artifacts"]["WAVECAR"]["exists"] is False
    assert result["artifacts"]["CHGCAR"]["size_bytes"] == 6
    assert result["files"]["out"]["truncated"] is True
    assert len(result["files"]["out"]["text"]) == 65536


def test_failed_legacy_return_includes_root_diagnostics(tmp_path, monkeypatch):
    state, h = setup_original(tmp_path)
    derived = prepare_recovery_handoff(state, h, CPU)
    old, remote = remote_copy(tmp_path, h, derived)
    task = recovery_plan(state, h)["tasks"][0]
    with zipfile.ZipFile(old / "inputs.zip") as archive:
        fake_result(old, task, archive)
    directory = old / task["result_directory"]
    result = json.loads((directory / "result.json").read_text())
    result.update(status="failed", error="vasp exited with code 1")
    (directory / "result.json").write_text(json.dumps(result))
    marker = json.loads((directory / "task.finished.json").read_text())
    marker.update(status="failed", result_checksum=file_checksum(directory / "result.json"))
    (directory / "task.finished.json").write_text(json.dumps(marker))
    seed_logs(old, task)
    monkeypatch.setenv("SLURM_JOB_ID", "fake-diagnostic")
    assert runner.main(remote) == 2  # Other tasks are still absent; never dispatch them.
    output = remote.with_name(remote.name[:-6] + "_output")
    with zipfile.ZipFile(output / "diagnostics.zip") as archive:
        report = json.loads(archive.read("recovery_status.json"))
    tid = task["expected"]["task_id"]
    assert report["science_submissions"] == 0
    assert "WAVECAR" in report["task_diagnostics"][tid]["files"]["failure_report.txt"]["text"]
    with zipfile.ZipFile(output / "recovered.results.zip") as archive:
        assert not any(n.endswith(("WAVECAR", "CHGCAR", "OUTCAR")) for n in archive.namelist())

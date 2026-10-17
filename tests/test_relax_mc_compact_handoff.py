import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from phase_agent.tools.relax_mc_batch import upload_driver as driver
from phase_agent.tools.relax_mc_batch.handoff import BOOTSTRAP, compact_handoff, result_archive
from phase_agent.tools.remote.transfer_bundle import create_bundle, input_files


@pytest.fixture
def tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp("n")


def package(root):
    root.mkdir()
    (root / "submit_all.sh").write_text(
        '#!/bin/bash\n#SBATCH --gres=gpu:1\nset -euo pipefail\ncd -- "${SLURM_SUBMIT_DIR:?x}"\npython3 run_batch.py\n'
    )
    (root / "inputs").mkdir()
    (root / "inputs/structure.vasp").write_text("structure")
    create_bundle(root, input_files(root), root / "inputs.zip", kind="inputs")
    return compact_handoff(root)


def test_named_three_files_manifest_resume_and_output(tmp_path, monkeypatch):
    root = tmp_path / "epoch0_RelaxMC_Batch01"
    folder = package(root)
    assert folder.name == root.name + "_input"
    assert {p.name for p in folder.iterdir()} == {
        "inputs.zip",
        "submit_all.sh",
        "run_upload.py",
    }
    assert (folder / "run_upload.py").read_text() == BOOTSTRAP
    script = (folder / "submit_all.sh").read_text()
    assert "python3 run_upload.py" in script and "python3 - <<" not in script
    assert script.index("SLURM_JOB_ID") < script.index("python3")
    calls = []

    def compute(*args, **kw):
        work = kw["cwd"]
        manifest = json.loads((work / "transfer_manifest.json").read_text())
        for row in manifest["files"]:
            assert hashlib.sha256((work / row["path"]).read_bytes()).hexdigest() == row["sha256"]
        calls.append(work)
        (work / "results.zip").write_bytes(b"fake scientific return")
        return 0

    monkeypatch.setenv("SLURM_JOB_ID", "fake-test-job")
    monkeypatch.setattr(driver.subprocess, "call", compute)
    assert driver.main(folder) == 0
    work = folder / "work"
    (work / "workflow_state.json").write_text("keep this progress")
    (work / "transfer_manifest.json").unlink()  # old broken unpacker
    assert driver.main(folder) == 0
    assert (work / "transfer_manifest.json").is_file()
    assert (work / "workflow_state.json").read_text() == "keep this progress"
    output = root.with_name(root.name + "_output")
    assert {p.name for p in output.iterdir()} == {
        "results.zip",
        "diagnostics.zip",
    }
    assert (
        json.loads(zipfile.ZipFile(output / "diagnostics.zip").read("run_status.json"))[
            "returncode"
        ]
        == 0
    )
    record = {
        "directory": str(root),
        "expected_results_archive": str(root / "returned/results.zip"),
    }
    assert result_archive(record) == output / "results.zip"
    from phase_agent.runtime.waiting_files_presentation import waiting_file_lines

    text = "\n".join(
        waiting_file_lines(
            {
                "tasks": [{"status": "pending", "relax_mc_bundle_id": "b"}],
                "relax_mc_batches": {"b": record},
            }
        )
    )
    assert str(folder) in text and str(output) in text


def test_no_python_execution_on_login_node(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="compute node"):
        driver.main(tmp_path / "batch_input")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name,tamper", [("../escape", False), ("safe", True)])
def test_corrupt_archive_publishes_diagnostic_output(tmp_path, monkeypatch, name, tamper):
    folder = tmp_path / "batch_input"
    folder.mkdir()
    payload = b"original"
    row = {
        "path": name,
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    with zipfile.ZipFile(folder / "inputs.zip", "w") as z:
        z.writestr(
            "transfer_manifest.json",
            json.dumps({"schema": 1, "kind": "inputs", "files": [row]}),
        )
        z.writestr(name, b"tampered" if tamper else payload)
    monkeypatch.setenv("SLURM_JOB_ID", "fake-test-job")
    assert driver.main(folder) == 1
    assert not (folder / "work").exists()
    output = tmp_path / "batch_output"
    status = json.loads(zipfile.ZipFile(output / "diagnostics.zip").read("run_status.json"))
    assert status["error"] and not status["results_available"]
    from phase_agent.tools.relax_mc_batch.results import recover_available

    state = {
        "relax_mc_batches": {
            "b": {
                "directory": str(tmp_path / "batch"),
                "expected_results_archive": str(output / "results.zip"),
            }
        }
    }
    recovered = recover_available(state, {})
    assert "results.zip" in recovered["relax_mc_batches"]["b"]["return_error"]


def test_manifest_tampering_rejected_and_legacy_return_supported(tmp_path, monkeypatch):
    root = tmp_path / "batch"
    folder = package(root)
    driver.prepare_work(folder)
    (folder / "work/transfer_manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="manifest changed"):
        driver.prepare_work(folder)
    legacy = root / "returned/results.zip"
    legacy.parent.mkdir()
    legacy.write_bytes(b"legacy")
    assert (
        result_archive({"directory": str(root), "expected_results_archive": str(legacy)}) == legacy
    )


def test_failure_return_preserved_in_output(tmp_path, monkeypatch):
    folder = package(tmp_path / "batch")
    monkeypatch.setenv("SLURM_JOB_ID", "fake-test-job")

    def compute(*args, **kw):
        (kw["cwd"] / "results.zip").write_bytes(b"failed scientific results")
        return 1

    monkeypatch.setattr(driver.subprocess, "call", compute)
    assert driver.main(folder) == 1
    output = tmp_path / "batch_output"
    assert (output / "results.zip").read_bytes() == b"failed scientific results"
    assert (
        json.loads(zipfile.ZipFile(output / "diagnostics.zip").read("run_status.json"))[
            "returncode"
        ]
        == 1
    )


def test_actual_coordinator_and_local_recovery_from_named_output(tmp_path, monkeypatch):
    """Exercise the real integrity preflight and return importer using fake science."""
    import importlib.util
    import phase_agent
    from phase_agent.tools.relax_mc_batch.run import run_batch
    from phase_agent.tools.relax_mc_batch.results import recover_available

    tests = Path(phase_agent.__file__).parents[1] / "tests/test_relax_mc_batch.py"
    spec = importlib.util.spec_from_file_location("relax_mc_test_helpers", tests)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    monkeypatch.setattr(
        "phase_agent.tools.remote.handoff_naming.batch_label",
        lambda *a, **k: "epoch0_RelaxMC_Batch01",
    )
    manager, state, config, context, action, result = helpers.package(tmp_path, count=3)
    executor, calls = helpers.fake_executor(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_ID", "fake-test-job")

    def compute(*args, **kw):
        outcome = run_batch(kw["cwd"], submit=True, batch_executor=executor)
        assert outcome["status"] in {"completed", "paused"}, outcome
        return 0

    monkeypatch.setattr(driver.subprocess, "call", compute)
    assert driver.main(result["upload_folder"]) == 0
    assert calls
    # Old metadata must also discover the newly downloaded output folder.
    incoming = result["state"]
    incoming["relax_mc_batches"][result["bundle_id"]]["expected_results_archive"] = str(
        Path(result["directory"]) / "returned/results.zip"
    )
    recovered = recover_available(incoming, context)
    record = recovered["relax_mc_batches"][result["bundle_id"]]
    assert record["status"] in {"completed", "paused"} and "return_error" not in record
    assert recovered["qbc_candidates"]

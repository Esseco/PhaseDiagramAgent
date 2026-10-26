"""Emitted HPC result helpers work with only the Python standard library."""

import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from phase_agent.tools.remote import transfer_bundle


def emitted_helper(root, producer):
    if producer == "manual":
        from phase_agent.tools.remote.manual_upload_runner import _portable_transfer_worker

        _portable_transfer_worker(root, "run_mlip_batch", "_handoff_mlip_batch")
        return root / "transfer_bundle.py"
    if producer == "gpu":
        from phase_agent.tools.remote.write_unix_shell_script import write_unix_shell_script

        write_unix_shell_script(root / "GPU.sh", "#!/bin/bash\n#SBATCH -N 1\ntrue\n")
        return root / "_gpu_transfer_bundle.py"
    if producer == "round":
        from phase_agent.tools.remote.submission_handoff import prepare_handoff
        from phase_agent.tools.remote.write_unix_shell_script import write_unix_shell_script

        directory = root / "submissions/epoch0"
        script = directory / "Relax-0001/inputs/batch/GPU.sh"
        write_unix_shell_script(script, "#!/bin/bash\n#SBATCH -N 1\ntrue\n")
        handoff = prepare_handoff(directory, [script])
        helper = root / "_round_transfer.py"
        with zipfile.ZipFile(handoff["upload_archive"]) as archive:
            helper.write_bytes(archive.read("_round_transfer.py"))
        return helper
    if producer == "legacy":
        from phase_agent.tools.remote.legacy_recovery import portable_transfer_source

        helper = root / "_transfer.py"
        helper.write_text(portable_transfer_source(), encoding="utf-8")
        return helper
    from phase_agent.configuration.defaults.default_mace_committee_config import (
        default_mace_committee_config,
    )
    from phase_agent.tools.workflows.prepare_remote_finetune import prepare_remote_finetune
    from tests.test_dft_comparison_csv import add_result

    state = {}
    for index in range(10):
        add_result(state, task_id=f"T{index}")
    settings = default_mace_committee_config()
    settings["labels"]["include_stress"] = False
    prepared = prepare_remote_finetune(
        state,
        {
            "mlip_finetune": settings,
            "upload_batches_directory": str(root),
            "mlip": {"version": "m1", "model_path": "/remote/m1.model"},
            "python_environments": {"remote_mlip": "mace"},
        },
    )
    assert prepared["status"] == "awaiting_remote_training"
    assert prepared["submitted"] is False
    job = next(iter(prepared["state"]["remote_finetune_jobs"].values()))
    return Path(job["directory"]) / "inputs/transfer_bundle.py"


@pytest.mark.parametrize("producer", ["manual", "gpu", "round", "legacy", "training"])
def test_emitted_helper_packages_and_consolidates_without_installed_app(tmp_path, producer):
    helper = emitted_helper(tmp_path, producer)
    results = tmp_path / "synthetic-results"
    task = results / "T1"
    task.mkdir(parents=True)
    (task / "result.json").write_text('{"task_id":"T1","status":"completed"}')
    (task / "task.finished.json").write_text('{"task_id":"T1","status":"completed"}')
    code = """
import importlib.util
import json
from pathlib import Path
import sys

assert importlib.util.find_spec("phase_agent") is None
spec = importlib.util.spec_from_file_location("portable_transfer", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
root = Path(sys.argv[2])
archive = module.result_bundle(root, [root / "T1"], "batch")
consolidated = module.consolidate_results(root)
assert module.validate_bundle(archive) == 2
assert module.validate_bundle(consolidated) == 2
print(json.dumps([module.MAX_ARCHIVE_FILES, module.MAX_MANIFEST_BYTES, module.MAX_ARCHIVE_BYTES]))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", code, str(helper), str(results)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == [
        transfer_bundle.MAX_ARCHIVE_FILES,
        transfer_bundle.MAX_MANIFEST_BYTES,
        transfer_bundle.MAX_ARCHIVE_BYTES,
    ]
    assert (task / "result.json").read_text() == '{"task_id":"T1","status":"completed"}'
    with zipfile.ZipFile(results / "transfer/all.results.zip") as archive:
        assert set(archive.namelist()) == {
            "transfer_manifest.json",
            "T1/result.json",
            "T1/task.finished.json",
        }


def test_saved_capacity_imports_reuse_the_authoritative_implementation():
    from phase_agent.tools.remote import transfer_limits

    assert transfer_limits.validate_capacity is transfer_bundle.validate_capacity
    constants = {}
    exec(transfer_limits.bootstrap_limits(), constants)
    for name in ("MAX_ARCHIVE_FILES", "MAX_MANIFEST_BYTES", "MAX_ARCHIVE_BYTES"):
        assert getattr(transfer_limits, name) == getattr(transfer_bundle, name) == constants[name]

"""Synthetic coverage of CPU-only recovery staging."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from phase_agent.tools.remote.integrity import file_checksum
from phase_agent.tools.remote.phase_cpu_recovery import stage_phase_cpu_recovery


def frozen_wave(tmp_path, monkeypatch):
    monkeypatch.setattr("phase_agent.tools.remote.input_validation.validate_handoff", lambda _: {})
    rows = [
        {
            "task_id": f"t{i}",
            "task_checksum": f"sha{i}",
            "result_path": f"calculations/refresh/results/{i}/result.json",
        }
        for i in range(201)
    ]
    raw = json.dumps({"tasks": rows}).encode()
    digest = hashlib.sha256(raw).hexdigest()
    parent = tmp_path / "source"
    parent.mkdir()
    archive = parent / "inputs.zip"
    directory = "calculations/refresh/inputs/zz-phase-analysis-batch"
    prefix = '#!/usr/bin/env bash\nset -euo pipefail\ncd -- "$(dirname "$0")"\nexport PYTHONPATH="$PWD/runtime:${PYTHONPATH:-}"\n'
    base = (
        "python3 -m phase_agent.tools.remote.phase_analysis_wave phase_wave.json ../../../.. --spec-checksum "
        + digest
    )
    cpu = "#SBATCH --array=0-1\n" + prefix + base + ' --group-index "$SLURM_ARRAY_TASK_ID"\n'
    finalize = prefix + base + " --finalize\n"
    finalize = finalize.replace(
        "set -euo pipefail",
        'PHASE_HANDOFF_ROOT="$(cd -- "../../../.." && pwd)"\ntrap \'python3 _round_transfer.py\' EXIT\nset -euo pipefail',
    )
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(directory + "/phase_wave.json", raw)
        z.writestr(directory + "/CPU.sh", cpu)
        z.writestr(directory + "/CPU_finalize.sh", finalize)
    return archive, file_checksum(archive), digest, {r["task_id"]: r["task_checksum"] for r in rows}


def test_recovery_preserves_frozen_scope_and_never_submits_gpu(tmp_path, monkeypatch):
    archive, sha, spec, tasks = frozen_wave(tmp_path, monkeypatch)
    root = "/scratch/project/epoch4/Batch03_input"
    out = tmp_path / "recovery"
    result = stage_phase_cpu_recovery(archive, sha, spec, tasks, root, out)
    assert result["task_count"] == 201 and result["group_count"] == 2
    assert result["result_roots"] == ["calculations/refresh/results"]
    cpu = (out / "CPU.sh").read_text()
    final = (out / "CPU_finalize.sh").read_text()
    assert "${SLURM_SUBMIT_DIR:?" in cpu and root in cpu
    assert root in final and "_round_transfer.py" in final
    assert sha in cpu and sha in final and "sha256sum -c -" in cpu and "sha256sum -c -" in final
    assert "GPU" not in cpu + final and "sbatch" not in cpu + final
    manifest = json.loads((out / "recovery_manifest.json").read_text())
    assert manifest["spec_sha256"] == spec
    assert file_checksum(out / "inputs.zip") == result["upload_archive_sha256"]
    with zipfile.ZipFile(out / "inputs.zip") as bundle:
        transfer = json.loads(bundle.read("transfer_manifest.json"))
        assert transfer["kind"] == "inputs"
        assert {entry["path"] for entry in transfer["files"]} == set(manifest["files"]) | {
            "recovery_manifest.json"
        }
    assert all(file_checksum(out / p) == h for p, h in manifest["files"].items())
    env = {**os.environ, "PYTHONPATH": str(out / "runtime"), "PYTHONNOUSERSITE": "1"}
    check = subprocess.run(
        [sys.executable, "-m", "phase_agent.tools.remote.phase_analysis_wave", "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert check.returncode == 0, check.stderr
    with pytest.raises(ValueError, match="destination_exists"):
        stage_phase_cpu_recovery(archive, sha, spec, tasks, root, out)


def test_recovery_rejects_mismatched_scope_and_bad_remote_path(tmp_path, monkeypatch):
    archive, sha, spec, tasks = frozen_wave(tmp_path, monkeypatch)
    out = tmp_path / "recovery"
    tasks["t1"] = "wrong"
    with pytest.raises(ValueError, match="task_identity_changed"):
        stage_phase_cpu_recovery(archive, sha, spec, tasks, "/scratch/Batch03_input", out)
    with pytest.raises(ValueError, match="invalid_remote_root"):
        stage_phase_cpu_recovery(archive, sha, spec, tasks, "/scratch/../Batch03_input", out)
    assert not out.exists()


def test_project_control_registers_and_reuses_without_replacing_original(tmp_path, monkeypatch):
    import shutil
    import threading
    from types import SimpleNamespace
    from phase_agent.runtime.phase_cpu_recovery_control import prepare

    monkeypatch.setattr(
        "phase_agent.tools.state.execution_receipts.recovery_report",
        lambda *args: {"unsettled": []},
    )
    source, sha, spec, checksums = frozen_wave(tmp_path, monkeypatch)
    project = tmp_path / "project"
    state_dir = project / "workflow_state"
    state_dir.mkdir(parents=True)
    epoch = project / "submissions" / "epoch4"
    folder = epoch / "delivery" / "transfers" / "Batch03_input"
    folder.mkdir(parents=True)
    archive = folder / "inputs.zip"
    shutil.copy2(source, archive)
    ids = list(checksums)
    handoff = {
        "upload_archive": str(archive),
        "import_directory": str(epoch),
        "return_directory": str(folder.with_name("Batch03_output")),
    }
    tasks = [
        {
            "task_id": tid,
            "task_checksum": digest,
            "status": "pending",
            "model_version": "m1",
            "config_version": "c1",
            "parent_decision_id": "approval",
        }
        for tid, digest in checksums.items()
    ]
    state = {
        "model_refresh": {"status": "waiting_results", "task_ids": ids, "approved_by": "approval"},
        "active_model_version": "m1",
        "confirmed_config_version": "c1",
        "tasks": tasks,
        "slurm_batches": [{"task_ids": ids, "handoff": handoff}],
    }
    state_path = state_dir / "state.json"
    state_path.write_text(json.dumps(state))
    control = SimpleNamespace(
        state_path=str(state_path), chat_handler=SimpleNamespace(lock=threading.Lock())
    )
    args = (control, sha, spec, len(ids), "/scratch/epoch4/Batch03_input")
    result = prepare(*args)
    assert result["status"] == "prepared" and not result["submitted"]
    assert Path(result["upload_archive"]).is_file()
    assert file_checksum(archive) == sha
    updated = json.loads(state_path.read_text())
    assert updated["tasks"] == tasks and updated["slurm_batches"][0]["handoff"] == handoff
    assert len(updated["phase_cpu_recoveries"]) == 1
    assert prepare(*args)["status"] == "reused"
    assert len(json.loads(state_path.read_text())["phase_cpu_recoveries"]) == 1

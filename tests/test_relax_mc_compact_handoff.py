import hashlib
import json
import subprocess
import zipfile

import pytest

from phase_agent.tools.relax_mc_batch.handoff import BOOTSTRAP, compact_handoff
from phase_agent.tools.remote.transfer_bundle import create_bundle, input_files


def test_three_file_handoff_and_verified_resume(tmp_path, monkeypatch):
    root = tmp_path / "batch"
    root.mkdir()
    (root / "submit_all.sh").write_text(
        '#!/bin/bash\n#SBATCH --gres=gpu:1\nset -euo pipefail\ncd -- "${SLURM_SUBMIT_DIR:?x}"\npython3 run_batch.py\n'
    )
    (root / "inputs").mkdir()
    (root / "inputs" / "structure.vasp").write_text("structure")
    create_bundle(root, input_files(root), root / "inputs.zip", kind="inputs")
    folder = compact_handoff(root)
    assert {p.name for p in folder.iterdir()} == {
        "inputs.zip",
        "submit_all.sh",
        "run_upload.py",
    }
    script = (folder / "submit_all.sh").read_text()
    assert "python3 run_upload.py" in script and "python3 - <<" not in script
    assert script.index("SLURM_JOB_ID") < script.index("python3")
    from phase_agent.runtime.waiting_files_presentation import waiting_file_lines

    state = {
        "tasks": [{"status": "pending", "relax_mc_bundle_id": "b"}],
        "relax_mc_batches": {
            "b": {
                "directory": str(root),
                "expected_results_archive": str(root / "returned/results.zip"),
            }
        },
    }
    text = "\n".join(waiting_file_lines(state))
    assert str(folder) in text and "work/results.zip" in text
    monkeypatch.chdir(folder)
    monkeypatch.setenv("SLURM_JOB_ID", "fake-test-job")
    calls = []
    monkeypatch.setattr(subprocess, "call", lambda *a, **kw: calls.append(kw) or 0)
    with pytest.raises(SystemExit) as done:
        exec(BOOTSTRAP, {})
    assert done.value.code == 0 and len(calls) == 1
    work = folder / "work"
    (work / "workflow_state.json").write_text("resume")
    (work / "inputs/structure.vasp").write_text("retained")
    with pytest.raises(SystemExit):
        exec(BOOTSTRAP, {})
    assert (work / "inputs/structure.vasp").read_text() == "retained"
    assert (work / "workflow_state.json").read_text() == "resume"
    assert calls[-1]["env"]["SLURM_SUBMIT_DIR"] == str(work.resolve())
    with (folder / "inputs.zip").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(RuntimeError, match="changed"):
        exec(BOOTSTRAP, {})


@pytest.mark.parametrize("name,tamper", [("../escape", False), ("safe", True)])
def test_bad_input_rejected_before_extraction(tmp_path, monkeypatch, name, tamper):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "fake-test-job")
    payload = b"original"
    row = {
        "path": name,
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    with zipfile.ZipFile("inputs.zip", "w") as z:
        z.writestr(
            "transfer_manifest.json",
            json.dumps({"schema": 1, "kind": "inputs", "files": [row]}),
        )
        z.writestr(name, b"tampered" if tamper else payload)
    with pytest.raises(ValueError):
        exec(BOOTSTRAP, {})
    assert not (tmp_path / "work").exists()


def test_python_refuses_login_node_before_unpacking(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="compute node"):
        exec(BOOTSTRAP, {})
    assert not (tmp_path / "work").exists()

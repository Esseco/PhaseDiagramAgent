import hashlib
from pathlib import Path
import pytest

from tests.test_dft_handoff_repair import fixture
from phase_agent.tools.remote.direct_dft_handoff_naming import name_direct_dft_handoff
from phase_agent.tools.remote.input_validation import validate_handoff


@pytest.fixture(autouse=True)
def valid_receipt(monkeypatch):
    from phase_agent.tools.workflows import direct_extrapolation_sampling
    monkeypatch.setattr(direct_extrapolation_sampling, "valid_direct_receipt", lambda receipt: True)


def setup(tmp_path):
    root, state, old, digest = fixture(tmp_path)
    state["direct_sampling_results"] = {"epoch_refresh": {"selected_count": 2, "selected": [
        {"candidate_id": t["object_id"], "structure_sha256": t["input_structure_sha256"]}
        for t in state["tasks"]]}}
    return root, state, old, digest


def test_named_handoff_preserves_input_and_records(tmp_path):
    root, state, old, digest = setup(tmp_path)
    before = [dict(t) for t in state["tasks"]]
    result = name_direct_dft_handoff(state, root, "decision", digest, "/remote/project")
    folder = Path(result["handoff"]["folder"])
    assert folder.name == "ExtrapolationRecovery_DFTSinglePoint_Batch01_input"
    assert hashlib.sha256((folder / "inputs.zip").read_bytes()).hexdigest() == digest
    assert validate_handoff(folder)["status"] == "passed"
    assert (folder / "README.md").is_file() and old.exists()
    assert state["tasks"] == before
    assert all(b["handoff"]["folder"] == str(folder) and b["superseded_handoffs"] for b in state["slurm_batches"])
    assert result["handoff"]["remote_input_directory"] == "/remote/project/epoch1/" + folder.name
    assert result["submitted"] is False
    with pytest.raises(ValueError, match="already_exists"):
        name_direct_dft_handoff(state, root, "decision", digest, "/remote/project")


@pytest.mark.parametrize("failure", ["submitted", "hash", "scope", "collision"])
def test_invalid_or_submitted_handoff_is_not_renamed(tmp_path, failure):
    root, state, old, digest = setup(tmp_path)
    if failure == "submitted":
        state["slurm_batches"][0]["job_id"] = "123"
    elif failure == "hash":
        digest = "0" * 64
    elif failure == "scope":
        state["tasks"][0]["input_structure_sha256"] = "wrong"
    else:
        (old.parent / "ExtrapolationRecovery_DFTSinglePoint_Batch01_input").mkdir()
    with pytest.raises(ValueError):
        name_direct_dft_handoff(state, root, "decision", digest, "/remote/project")
    assert old.is_dir()
    assert all(b["handoff"]["folder"] == str(old) for b in state["slurm_batches"])


def test_future_handoff_name_distinguishes_direct_from_ordinary(tmp_path):
    import json
    from phase_agent.tools.remote.handoff_naming import handoff_label

    root = tmp_path / "epoch2"
    job = root / "inputs" / "job"
    job.mkdir(parents=True)
    (job / "GPU.sh").write_text("#!/bin/bash\n", encoding="utf8")
    (job / "manifest.json").write_text(json.dumps([{"task_id": "t", "stage": "dft_single_point"}]), encoding="utf8")
    task = job / "task.json"
    task.write_text(json.dumps({"task_id": "t", "object_id": "DIRECT-abc"}), encoding="utf8")
    assert handoff_label(root, ["inputs/job/GPU.sh"]) == "epoch2_ExtrapolationRecoveryDFTSinglePoint"
    task.write_text(json.dumps({"task_id": "t", "object_id": "S-abc"}), encoding="utf8")
    assert handoff_label(root, ["inputs/job/GPU.sh"]) == "epoch2_DFTSinglePoint"

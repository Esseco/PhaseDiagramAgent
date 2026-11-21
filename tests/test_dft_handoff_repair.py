import json
from hashlib import sha256
from pathlib import Path
import zipfile

import pytest
import threading
from types import SimpleNamespace

from phase_agent.tools.remote.dft_handoff_repair import repair_dft_handoff
from phase_agent.tools.remote.input_validation import validate_handoff
from phase_agent.tools.remote.transfer_bundle import create_bundle, input_files


def fixture(tmp_path):
    root = tmp_path / "project"
    source = tmp_path / "payload"
    source.mkdir()
    output = root / "submissions" / "epoch1" / "delivery" / "transfers" / "DFTSP_Batch01_input"
    output.mkdir(parents=True)
    geom = b"Si\n1\n1 0 0\n0 1 0\n0 0 1\nSi\n1\nDirect\n0 0 0\n"
    digest = sha256(geom).hexdigest()
    script = b"#!/bin/bash\n#SBATCH -n 2\n#SBATCH --ntasks-per-node=2\n#SBATCH --gres=gpu:1\n#SBATCH --partition=v100m3\npython3 worker.py\n"
    tasks, batches, scripts, decisions = [], [], [], []
    for n in range(2):
        task_id, object_id = f"D{n}", f"S:frame:{n}"
        task = {
            "task_id": task_id,
            "object_id": object_id,
            "structure_id": "S",
            "status": "pending",
            "stage": "dft_single_point",
            "parent_decision_id": "decision",
            "task_key": f"key{n}",
            "model_version": "m",
            "config_version": "c",
            "input_structure_sha256": digest,
        }
        folder = source / "calculations" / f"job{n}"
        folder.mkdir(parents=True)
        (folder / "GPU.sh").write_bytes(script)
        (folder / "submit_gpu.sh").write_bytes(script)
        (folder / "worker.py").write_text("print('ok')\n", encoding="utf8")
        (folder / "task.json").write_text(json.dumps(task), encoding="utf8")
        (folder / "initial_structure.vasp").write_bytes(geom)
        scripts.append((folder / "GPU.sh").relative_to(source).as_posix())
        tasks.append(task)
        decisions.append({"candidate_id": object_id, "action": "DFT_SINGLE_POINT"})
    (source / "submission_plan.json").write_text(json.dumps({"scripts": scripts}), encoding="utf8")
    create_bundle(source, input_files(source), output / "inputs.zip", kind="inputs")
    (output / "submit_all.sh").write_text(
        "#!/bin/bash\npython3 dispatch_tasks.py .\n", encoding="utf8", newline="\n"
    )
    (output / "dispatch_tasks.py").write_text("print('ok')\n", encoding="utf8")
    handoff = {
        "folder": str(output),
        "upload_archive": str(output / "inputs.zip"),
        "batch_ids": ["B0", "B1"],
        "job_count": 2,
    }
    batches = [
        {
            "batch_id": f"B{n}",
            "status": "prepared",
            "job_id": None,
            "task_ids": [f"D{n}"],
            "handoff": dict(handoff),
        }
        for n in range(2)
    ]
    state = {
        "active_model_version": "m",
        "confirmed_config_version": "c",
        "action_records": [
            {
                "record_id": "decision",
                "status": "prepared",
                "human_feedback": {"decision": "approve"},
                "final_action": {
                    "tool": "select_dft_candidates",
                    "parameters": {"decisions": decisions},
                },
            }
        ],
        "tasks": tasks,
        "slurm_batches": batches,
    }
    return root, state, output, sha256((output / "inputs.zip").read_bytes()).hexdigest()


def test_repair_preserves_identity_and_repackages_one_rank(tmp_path):
    root, state, old, digest = fixture(tmp_path)
    result = repair_dft_handoff(state, root, "decision", digest, 2)
    new = Path(result["handoff"]["folder"])
    assert new.name == "DFTSP_Batch02_input"
    assert old.is_dir() and validate_handoff(new)["status"] == "passed"
    assert result["task_count"] == 2 and result["submitted"] is False
    assert all(row["handoff"]["folder"] == str(new) for row in state["slurm_batches"])
    again = repair_dft_handoff(state, root, "decision", digest, 2)
    assert again["status"] == "reused" and again["handoff"]["folder"] == str(new)
    with (
        zipfile.ZipFile(old / "inputs.zip") as before,
        zipfile.ZipFile(new / "inputs.zip") as after,
    ):
        for name in before.namelist():
            if name == "transfer_manifest.json":
                continue
            if name.endswith(("/GPU.sh", "/submit_gpu.sh")):
                assert b"#SBATCH -n 1\n" in after.read(name)
                assert b"#SBATCH --ntasks-per-node=1\n" in after.read(name)
            else:
                assert before.read(name) == after.read(name)


@pytest.mark.parametrize("failure", ["submitted", "scope", "digest", "mixed", "model"])
def test_repair_rejects_unverified_state_without_new_package(tmp_path, failure):
    root, state, old, digest = fixture(tmp_path)
    if failure == "submitted":
        state["slurm_batches"][0]["job_id"] = "123"
    if failure == "scope":
        state["tasks"][0]["object_id"] = "wrong"
    if failure == "digest":
        digest = "0" * 64
    if failure == "mixed":
        state["slurm_batches"][0]["task_ids"].append("other")
    if failure == "model":
        state["active_model_version"] = "other"
    with pytest.raises(ValueError):
        repair_dft_handoff(state, root, "decision", digest, 2)
    assert not (old.parent / "DFTSP_Batch02_input").exists()


def test_project_control_persists_repaired_handoff_without_changing_tasks(tmp_path, monkeypatch):
    from phase_agent.runtime.local_agent_control import LocalAgentControl
    import phase_agent.tools.state.execution_receipts as receipts

    root, state, old, digest = fixture(tmp_path)
    path = root / "workflow_state" / "state.json"
    path.parent.mkdir()
    path.write_text(json.dumps(state), encoding="utf8")
    monkeypatch.setattr(receipts, "recovery_report", lambda *_: {"unsettled": []})
    control = LocalAgentControl(SimpleNamespace(state_path=path, lock=threading.RLock()))
    result = control.repair_dft_handoff("decision", digest, 2)
    saved = json.loads(path.read_text(encoding="utf8"))
    assert result["status"] == "prepared"
    assert saved["tasks"] == state["tasks"]
    assert saved["action_records"] == state["action_records"]
    assert all(
        b["handoff"]["folder"] == result["handoff"]["folder"] for b in saved["slurm_batches"]
    )
    assert len(saved["dft_handoff_repairs"]) == 1
    assert control.repair_dft_handoff("decision", digest, 2)["status"] == "reused"
    assert len(json.loads(path.read_text(encoding="utf8"))["dft_handoff_repairs"]) == 1


def test_project_control_blocks_unsettled_work(tmp_path, monkeypatch):
    from phase_agent.runtime.local_agent_control import LocalAgentControl
    import phase_agent.tools.state.execution_receipts as receipts

    root, state, old, digest = fixture(tmp_path)
    path = root / "workflow_state" / "state.json"
    path.parent.mkdir()
    original = json.dumps(state)
    path.write_text(original, encoding="utf8")
    monkeypatch.setattr(receipts, "recovery_report", lambda *_: {"unsettled": ["uncertain"]})
    control = LocalAgentControl(SimpleNamespace(state_path=path, lock=threading.RLock()))
    with pytest.raises(ValueError, match="pending_or_unsettled"):
        control.repair_dft_handoff("decision", digest, 2)
    assert path.read_text(encoding="utf8") == original


def test_split_repair_preserves_tasks_and_partitions_every_script(tmp_path):
    root, state, old, digest = fixture(tmp_path)
    original_tasks = json.loads(json.dumps(state["tasks"]))
    result = repair_dft_handoff(state, root, "decision", digest, 2,
                                partition_counts={"v100m3": 1, "v100g32": 1})
    assert state["tasks"] == original_tasks
    assert result["handoff"]["partition_counts"] == {"v100m3": 1, "v100g32": 1}
    new = Path(result["handoff"]["folder"])
    assert validate_handoff(new)["status"] == "passed"
    with zipfile.ZipFile(new / "inputs.zip") as z:
        scripts = json.loads(z.read("submission_plan.json"))["scripts"]
        for path, partition in zip(scripts, ("v100m3", "v100g32"), strict=True):
            assert ("#SBATCH --partition=" + partition).encode() in z.read(path)
            assert ("#SBATCH --partition=" + partition).encode() in z.read(path.rsplit("/",1)[0] + "/submit_gpu.sh")
    assert repair_dft_handoff(state, root, "decision", digest, 2,
                              partition_counts={"v100m3": 1, "v100g32": 1})["status"] == "reused"
    with pytest.raises(ValueError, match="partition_counts_changed"):
        repair_dft_handoff(state, root, "decision", digest, 2, partition_counts={"v100m3": 2})

@pytest.mark.parametrize("counts", [{"v100m3": 3}, {"v100m3": 1, "bad shell;": 1}, {"v100m3": -1, "v100g32": 3}])
def test_split_repair_rejects_bad_partition_allocation(tmp_path, counts):
    root, state, old, digest = fixture(tmp_path)
    with pytest.raises(ValueError, match="invalid_partition_counts"):
        repair_dft_handoff(state, root, "decision", digest, 2, partition_counts=counts)
    assert not (old.parent / "DFTSP_Batch02_input").exists()

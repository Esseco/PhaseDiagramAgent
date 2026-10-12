import json
import shutil
import sys
import types
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from phase_agent.tools.remote.submission_handoff import (
    DRIVER,
    ensure_waiting_handoffs,
    prepare_handoff,
)
from phase_agent.tools.remote.transfer_bundle import (
    has_registered_return,
    import_registered_bundles,
    publish_return,
    registered_result_directories,
)


def setup_round(tmp_path):
    root = tmp_path / "submissions/epoch1_model"
    batches, tasks = [], []
    for op, name in [("SP", "one"), ("SP", "two"), ("Relax", "three")]:
        directory = root / op / "inputs" / name
        directory.mkdir(parents=True)
        script = directory / "GPU.sh"
        script.write_text("#!/bin/bash\n#SBATCH --gres=gpu:1\npython3 run.py\n")
        (directory / "run.py").write_text('print("fake")')
        batches.append(
            {
                "batch_id": name,
                "task_ids": [name],
                "status": "prepared",
                "upload_directory": str(directory),
                "script_path": str(script),
                "results_directory": str(root / op / "results"),
            }
        )
        tasks.append({"task_id": name, "status": "pending"})
    return root, {"slurm_batches": batches, "tasks": tasks}


def extract(handoff, remote):
    with zipfile.ZipFile(handoff["upload_archive"]) as archive:
        archive.extractall(remote)


def completed(root, task):
    root.mkdir(parents=True, exist_ok=True)
    directory = root / task
    directory.mkdir(exist_ok=True)
    (directory / "result.json").write_text(json.dumps({"task_id": task, "energy": -1}))
    (directory / "task.finished.json").write_text('{"status":"completed"}')
    return directory


def test_one_folder_dispatch_and_automatic_round_return(tmp_path):
    root, state = setup_round(tmp_path)
    ensure_waiting_handoffs(state)
    handoff = state["slurm_batches"][0]["handoff"]
    assert all(row["handoff"] == handoff for row in state["slurm_batches"])
    assert handoff["job_count"] == 3
    assert {p.name for p in Path(handoff["folder"]).iterdir()} == {
        "inputs.zip",
        "dispatch_tasks.py",
        "submit_all.sh",
    }
    remote = tmp_path / "remote"
    extract(handoff, remote)
    for script in json.loads((remote / "submission_plan.json").read_text())["scripts"]:
        text = (remote / script).read_text()
        assert "trap " in text and "_round_transfer.py" in text
        assert text.index("#SBATCH") < text.index("trap") < text.index("python3 run.py")
    completed(remote / "SP/results", "one")
    completed(remote / "Relax/results", "three")
    publish_return(remote, "SP/results")
    publish_return(remote, "Relax/results")
    shutil.copytree(remote / "return-results", Path(handoff["return_directory"]))
    assert has_registered_return(state)
    assert import_registered_bundles(registered_result_directories(state)) == []
    assert (root / "SP/results/one/result.json").is_file()
    assert (root / "Relax/results/three/result.json").is_file()
    assert not has_registered_return(state)
    before = Path(handoff["upload_archive"]).stat().st_mtime_ns
    ensure_waiting_handoffs(state)
    assert Path(handoff["upload_archive"]).stat().st_mtime_ns == before


def test_fake_sbatch_dispatch_retries_do_not_duplicate(tmp_path, monkeypatch):
    _, state = setup_round(tmp_path)
    ensure_waiting_handoffs(state)
    h = state["slurm_batches"][0]["handoff"]
    remote = tmp_path / "remote"
    remote.mkdir()
    shutil.copy2(h["upload_archive"], remote / "inputs.zip")
    monkeypatch.setenv("SLURM_JOB_ID", "999")
    monkeypatch.setitem(
        sys.modules, "fcntl", types.SimpleNamespace(flock=lambda *args: None, LOCK_EX=1, LOCK_NB=2)
    )
    monkeypatch.setattr(shutil, "which", lambda name: "fake-sbatch")
    import subprocess

    calls = []

    def run(cmd, **kwargs):
        calls.append((cmd, kwargs["cwd"]))
        return types.SimpleNamespace(returncode=0, stdout=str(100 + len(calls)), stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    namespace = {"__name__": "test_driver"}
    exec(DRIVER, namespace)
    namespace["run"](remote)
    namespace["run"](remote)
    assert len(calls) == 3
    assert all(cmd == ["sbatch", "--parsable", "GPU.sh"] for cmd, _ in calls)


def test_parallel_publish_and_corruption_rejected(tmp_path):
    remote = tmp_path / "remote"
    for i in range(4):
        completed(remote / "SP/results", str(i))
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(pool.map(lambda _: publish_return(remote, "SP/results"), range(4)))
    package = remote / "return-results/SP__results.results.zip"
    with zipfile.ZipFile(package) as archive:
        assert archive.testzip() is None
        assert all(f"SP/results/{i}/result.json" in archive.namelist() for i in range(4))
        content = {name: archive.read(name) for name in archive.namelist()}
    content["SP/results/0/result.json"] = b"corrupt"
    local = tmp_path / "local"
    (local / "return-results").mkdir(parents=True)
    with zipfile.ZipFile(local / "return-results/bad.results.zip", "w") as archive:
        for name, data in content.items():
            archive.writestr(name, data)
    errors = import_registered_bundles([local])
    assert len(errors) == 1
    assert not (local / "SP/results/0/result.json").exists()


def test_existing_published_batches_not_repackaged(tmp_path):
    root, state = setup_round(tmp_path)
    first = state["slurm_batches"][0]
    first["handoff"] = prepare_handoff(root / "SP", [first["script_path"]])
    old = first["handoff"].copy()
    ensure_waiting_handoffs(state)
    assert first["handoff"] == old
    assert state["slurm_batches"][1]["handoff"]["job_count"] == 2


def test_index_handles_resolved_archive_alias(tmp_path, monkeypatch):
    from phase_agent.tools.remote.submission_layout import write_submission_index

    root, state = setup_round(tmp_path)
    archive = tmp_path / "diagrams/diagram_default/submissions/archives/old.results.zip"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"archive")
    monkeypatch.setattr(
        "phase_agent.tools.remote.submission_storage.storage_archive", lambda *args: archive
    )
    index = write_submission_index(root.parent, state)
    assert "old.results.zip" in index.read_text(encoding="utf-8")


def test_dispatcher_refuses_login_node_before_writing(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    namespace = {"__name__": "test_driver"}
    exec(DRIVER, namespace)
    with pytest.raises(RuntimeError, match="Submit with sbatch"):
        namespace["run"](tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_outer_script_is_slurm_job(tmp_path):
    _, state = setup_round(tmp_path)
    ensure_waiting_handoffs(state)
    handoff = state["slurm_batches"][0]["handoff"]
    text = Path(handoff["submit_script"]).read_text()
    assert handoff["command"] == "sbatch submit_all.sh"
    assert "#SBATCH --job-name=phase-dispatch" in text
    assert "SLURM_SUBMIT_DIR" in text
    assert "python3 dispatch_tasks.py" in text
    assert "PHASE_SUBMIT_PY" not in text

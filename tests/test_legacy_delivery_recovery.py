"""Real original selections; fake terminal data in private copies only."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

import pytest
from pymatgen.core import Structure
from types import SimpleNamespace

from phase_agent.tools.remote.legacy_recovery import prepare_recovery_handoff, recovery_plan
from phase_agent.tools.remote import legacy_recovery_runner as runner
from phase_agent.tools.remote.transfer_bundle import (
    registered_return_locations,
    has_registered_return,
    import_registered_bundles,
    create_bundle,
)
from phase_agent.tools.remote.integrity import file_checksum

CPU = {
    "slurm_options": {"partition": "p1", "nodes": 1, "ntasks": 1, "mem": "2G", "time": "00:20:00"}
}


@pytest.fixture
def tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp("c")


def relocated(value, original, replacement):
    if isinstance(value, dict):
        return {k: relocated(v, original, replacement) for k, v in value.items()}
    if isinstance(value, list):
        return [relocated(v, original, replacement) for v in value]
    return (
        value.replace(str(original), str(replacement)).replace(
            original.as_posix(), replacement.as_posix()
        )
        if isinstance(value, str)
        else value
    )


def setup_original(tmp_path, project="O3-1ele-NaCrO2"):
    original = Path("E:/0-PhaseDiagram") / project
    if not (original / "workflow_state/state.json").is_file():
        pytest.skip("read-only actual project fixture is unavailable")
    saved = json.loads((original / "workflow_state/state.json").read_text(encoding="utf-8"))
    handoff = next(
        b["handoff"]
        for b in saved["slurm_batches"]
        if b.get("handoff") and Path(b["handoff"]["folder"]).name == "epoch0_DFTSP_Batch01"
    )
    selected = [
        b
        for b in saved["slurm_batches"]
        if (b.get("handoff") or {}).get("folder") == handoff["folder"]
    ]
    ids = {i for b in selected for i in b["task_ids"]}
    local = tmp_path / "local"
    state = relocated(
        {"tasks": [t for t in saved["tasks"] if t["task_id"] in ids], "slurm_batches": selected},
        original,
        local,
    )
    h = state["slurm_batches"][0]["handoff"]
    folder = Path(h["folder"])
    folder.mkdir(parents=True)
    for name in ("inputs.zip", "submit_all.sh", "dispatch_tasks.py"):
        shutil.copy2(Path(handoff["folder"]) / name, folder / name)
    with zipfile.ZipFile(h["upload_archive"]) as z:
        for task in state["tasks"]:
            relative = Path(task["input_path"]).relative_to(Path(h["import_directory"])).as_posix()
            target = Path(task["input_path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(relative))
    return state, h


def remote_copy(tmp_path, h, derived):
    old = tmp_path / "hpc" / Path(h["folder"]).name
    old.mkdir(parents=True)
    shutil.copy2(h["upload_archive"], old / "inputs.zip")
    new = old.with_name(Path(derived["folder"]).name)
    shutil.copytree(derived["folder"], new)
    return old, new


def fake_result(old, task, zipped, *, worker=False):
    from phase_agent.science.dft.vasp_training_labels import final_training_labels
    from phase_agent.science.structures.load_result_structure import load_result_structure

    directory = old / (task["input_directory"] if worker else task["result_directory"])
    directory.mkdir(parents=True, exist_ok=True)
    structure = Structure.from_str(
        zipped.read(task["input_directory"] + "/initial_structure.vasp").decode(), fmt="poscar"
    )
    labels = final_training_labels(
        SimpleNamespace(
            converged=True,
            ionic_steps=[
                {
                    "structure": structure,
                    "e_0_energy": -float(len(structure)),
                    "forces": [[0.0, 0.0, 0.0]] * len(structure),
                }
            ],
        )
    )
    (directory / "training.json").write_text(json.dumps(labels))
    outputs = {
        **labels,
        "energy": -float(len(structure)),
        "energy_unit": "eV",
        "composition": structure.composition.as_dict(),
        "training_file": "training.json",
        "training_checksum": file_checksum(directory / "training.json"),
    }
    _, digest = load_result_structure(outputs)
    prediction = {
        **{k: task["expected"][k] for k in ["task_id", "task_key", "model_version"]},
        "status": "completed",
        "model_sha256": "c" * 64,
        "structure_sha256": digest,
        "final_frame_index": outputs.get("final_frame_index"),
        "energy": -0.99 * len(structure),
        "energy_unit": "eV",
        "forces_unit": "eV/angstrom",
        "forces": [[0.0, 0.0, 0.0]] * len(structure),
    }
    (directory / "mlip_result.json").write_text(json.dumps(prediction))
    outputs.update(
        mlip_result_file="mlip_result.json",
        mlip_result_checksum=file_checksum(directory / "mlip_result.json"),
    )
    data = {
        **task["expected"],
        "stage": "dft_single_point",
        "status": "completed",
        "converged": True,
        "outputs": outputs,
    }
    (directory / "result.json").write_text(json.dumps(data))
    (directory / "task.finished.json").write_text(
        json.dumps(
            {
                **task["expected"],
                "status": "completed",
                "result_checksum": file_checksum(directory / "result.json"),
            }
        )
    )
    # These must never be transported.
    (directory / "OUTCAR").write_bytes(b"large remote-only log")
    (directory / "POTCAR").write_bytes(b"never transfer")
    return directory


def frozen(root):
    return {
        p.relative_to(root).as_posix(): file_checksum(p) for p in root.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize("project,count", [("O3-1ele-NaCoO2", 35), ("O3-1ele-NaCrO2", 15)])
def test_original_full_batch_collection_analysis_idempotence(tmp_path, monkeypatch, project, count):
    from phase_agent.tools.remote.batch_runner import RemoteBatchRunner
    from phase_agent.runtime.waiting_files_presentation import waiting_file_lines

    state, h = setup_original(tmp_path, project)
    original = deepcopy(state)
    old_local = frozen(Path(h["folder"]))
    derived = prepare_recovery_handoff(state, h, CPU)
    assert len(derived["task_ids"]) == count
    assert {p.name for p in Path(derived["folder"]).iterdir()} == {
        "inputs.zip",
        "submit_all.sh",
        "recover_results.py",
    }
    assert prepare_recovery_handoff(state, h, CPU) == derived
    old, remote = remote_copy(tmp_path, h, derived)
    plan = recovery_plan(state, h)
    with zipfile.ZipFile(old / "inputs.zip") as z:
        for index, task in enumerate(plan["tasks"]):
            fake_result(old, task, z, worker=index % 2 == 0)
    (old / ".submitted_jobs.json").write_text(
        json.dumps(
            {
                "archive_sha256": file_checksum(old / "inputs.zip"),
                "jobs": {"original-job": {"status": "submitted", "job_id": "12345"}},
            }
        )
    )
    source_before = frozen(old)
    monkeypatch.setenv("SLURM_JOB_ID", "fake-recovery-cpu")
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: pytest.fail("No scheduler/scientific subprocess allowed")
    )
    monkeypatch.setattr(
        subprocess, "Popen", lambda *a, **k: pytest.fail("No scientific process allowed")
    )
    assert runner.main(remote) == 0
    assert frozen(old) == source_before
    output = remote.with_name(remote.name[:-6] + "_output")
    assert {p.name for p in output.iterdir()} == {"recovered.results.zip", "diagnostics.zip"}
    with zipfile.ZipFile(output / "recovered.results.zip") as z:
        assert sum(n.endswith("/result.json") for n in z.namelist()) == count
        assert not any(n.endswith(("OUTCAR", "POTCAR", ".sh", ".py", ".zip")) for n in z.namelist())
    shutil.copytree(output, derived["return_directory"], dirs_exist_ok=True)
    assert has_registered_return(state)
    text = "\n".join(waiting_file_lines(state))
    assert "仅回收" in text and derived["folder"] in text and "独立目录" not in text
    collected = RemoteBatchRunner(
        tmp_path / "unused", worker_command=[]
    ).collect_results_with_report(state)
    assert collected["report"]["transfer_errors"] == [], collected["report"]
    assert collected["report"]["recovered_count"] == count, collected["report"]
    assert len(collected["results"]) == count
    assert all(r["outputs"].get("remote_mlip_prediction") for r in collected["results"])
    assert not has_registered_return(state)
    assert import_registered_bundles([], locations=registered_return_locations(state)) == []
    assert state == original and frozen(Path(h["folder"])) == old_local
    assert runner.main(remote) == 0
    assert frozen(old) == source_before


def test_partial_running_retry_and_unknown_submission_never_dispatch(tmp_path, monkeypatch):
    state, h = setup_original(tmp_path)
    d = prepare_recovery_handoff(state, h, CPU)
    old, remote = remote_copy(tmp_path, h, d)
    plan = recovery_plan(state, h)
    monkeypatch.setenv("SLURM_JOB_ID", "fake-cpu")
    assert runner.main(remote) == 2
    with zipfile.ZipFile(remote.with_name(remote.name[:-6] + "_output") / "diagnostics.zip") as z:
        status = json.loads(z.read("recovery_status.json"))
    assert status["science_submissions"] == 0 and len(status["missing_task_ids"]) == 15
    assert "unknown" in status["original_submission_states"]
    with zipfile.ZipFile(old / "inputs.zip") as z:
        fake_result(old, plan["tasks"][0], z, worker=True)
    assert runner.main(remote) == 2
    output = remote.with_name(remote.name[:-6] + "_output")
    shutil.copytree(output, d["return_directory"], dirs_exist_ok=True)
    errors = import_registered_bundles([], locations=registered_return_locations(state))
    assert len(errors) == 1 and "collection_incomplete" in errors[0]["reason"]
    with zipfile.ZipFile(old / "inputs.zip") as z:
        for task in plan["tasks"][1:]:
            fake_result(old, task, z)
    assert runner.main(remote) == 0
    shutil.copytree(output, d["return_directory"], dirs_exist_ok=True)
    assert import_registered_bundles([], locations=registered_return_locations(state)) == []
    assert not has_registered_return(state)


def test_tampering_and_outside_scope_are_rejected(tmp_path, monkeypatch):
    state, h = setup_original(tmp_path)
    d = prepare_recovery_handoff(state, h, CPU)
    old, remote = remote_copy(tmp_path, h, d)
    plan = recovery_plan(state, h)
    monkeypatch.setenv("SLURM_JOB_ID", "fake-cpu")
    with zipfile.ZipFile(old / "inputs.zip") as z:
        directory = fake_result(old, plan["tasks"][0], z)
    (directory / "training.json").write_text("{}")
    assert runner.main(remote) == 2
    output = remote.with_name(remote.name[:-6] + "_output")
    assert not (output / "recovered.results.zip").exists()
    (old / "inputs.zip").write_bytes(b"changed original")
    assert runner.main(remote) == 2
    with zipfile.ZipFile(output / "diagnostics.zip") as z:
        assert "original_input_archive_changed" in z.read("recovery_status.json").decode()
    # A correctly bound ZIP still cannot write outside the original task scope.
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "unrelated.json").write_text("{}")
    create_bundle(
        scratch,
        [scratch / "unrelated.json"],
        Path(d["return_directory"]) / "bad.results.zip",
        kind="results",
        binding={
            "batch": Path(d["folder"]).name[:-6],
            "input_sha256": file_checksum(d["upload_archive"]),
        },
    )
    errors = import_registered_bundles([], locations=registered_return_locations(state))
    assert "recovery_result_outside_original_scope" in errors[0]["reason"]
    assert not (Path(h["import_directory"]) / "unrelated.json").exists()


def test_login_node_refuses_before_any_write(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="CPU compute node"):
        runner.main(tmp_path / "x_input")
    assert not list(tmp_path.iterdir())


def test_recovery_presentation_with_unrelated_pending_task(tmp_path):
    from phase_agent.runtime.waiting_files_presentation import waiting_file_lines

    state, h = setup_original(tmp_path)
    derived = prepare_recovery_handoff(state, h, CPU)
    state["tasks"].append(
        {
            "task_id": "unrelated",
            "stage": "dft_single_point",
            "status": "pending",
            "input_path": str(tmp_path / "unrelated/task.json"),
        }
    )
    text = "\n".join(waiting_file_lines(state))
    assert derived["folder"] in text and "仅回收" in text


def test_changed_plan_cannot_be_discovered_as_registered_delivery(tmp_path):
    from phase_agent.tools.remote.legacy_recovery import recovery_handoffs

    state, h = setup_original(tmp_path)
    d = prepare_recovery_handoff(state, h, CPU)
    with zipfile.ZipFile(d["upload_archive"]) as z:
        payloads = {name: z.read(name) for name in z.namelist() if name != "transfer_manifest.json"}
    plan = json.loads(payloads["recovery_plan.json"])
    plan["tasks"][0]["expected"]["task_id"] = "forged"
    payloads["recovery_plan.json"] = json.dumps(plan).encode()
    source = tmp_path / "forged"
    source.mkdir()
    for name, data in payloads.items():
        (source / name).write_bytes(data)
    create_bundle(source, list(source.iterdir()), d["upload_archive"], kind="inputs")
    with pytest.raises(ValueError, match="outside_original_scope"):
        list(recovery_handoffs(state))


def test_missing_completion_and_conflicting_results_do_not_trigger_recomputation(
    tmp_path, monkeypatch
):
    state, h = setup_original(tmp_path)
    d = prepare_recovery_handoff(state, h, CPU)
    old, remote = remote_copy(tmp_path, h, d)
    plan = recovery_plan(state, h)
    task = plan["tasks"][0]
    monkeypatch.setenv("SLURM_JOB_ID", "fake-cpu")
    with zipfile.ZipFile(old / "inputs.zip") as z:
        canonical = fake_result(old, task, z)
        worker = fake_result(old, task, z, worker=True)
    (canonical / "task.finished.json").unlink()
    assert runner.main(remote) == 2  # The verified worker may still be recovered.
    (worker / "task.finished.json").unlink()
    assert runner.main(remote) == 2
    assert not (remote.with_name(remote.name[:-6] + "_output") / "recovered.results.zip").exists()


def test_shipped_entry_runs_in_isolated_python_without_installed_agent(tmp_path):
    import os
    import sys

    state, h = setup_original(tmp_path)
    d = prepare_recovery_handoff(state, h, CPU)
    old, remote = remote_copy(tmp_path, h, d)
    plan = recovery_plan(state, h)
    with zipfile.ZipFile(old / "inputs.zip") as z:
        fake_result(old, plan["tasks"][0], z, worker=True)
    env = {**os.environ, "SLURM_JOB_ID": "fake-collection-only", "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(
        [sys.executable, "-I", str(remote / "recover_results.py")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 2, result.stderr
    report = json.loads(result.stdout)
    assert report["science_submissions"] == 0 and len(report["recovered_task_ids"]) == 1
    with zipfile.ZipFile(
        remote.with_name(remote.name[:-6] + "_output") / "recovered.results.zip"
    ) as z:
        assert any(n.endswith("/training.json") for n in z.namelist())


def test_original_archive_and_identity_are_immutable_when_deriving_input(tmp_path):
    state, h = setup_original(tmp_path)
    before = deepcopy(state)
    old = frozen(Path(h["folder"]))
    derived = prepare_recovery_handoff(state, h, CPU)
    with zipfile.ZipFile(derived["upload_archive"]) as z:
        plan = json.loads(z.read("recovery_plan.json"))
        assert {r["expected"]["task_id"] for r in plan["tasks"]} == {
            t["task_id"] for t in state["tasks"]
        }
        assert z.namelist() == ["_transfer.py", "recovery_plan.json", "transfer_manifest.json"]
    assert state == before and frozen(Path(h["folder"])) == old


def test_explicit_renamed_original_directory_remains_bound_to_old_archive(tmp_path, monkeypatch):
    state, h = setup_original(tmp_path)
    d = prepare_recovery_handoff(state, h, CPU)
    old, remote = remote_copy(tmp_path, h, d)
    renamed = old.with_name("original-folder-was-renamed")
    assert old.resolve().is_relative_to(tmp_path.resolve()) and renamed.resolve().is_relative_to(
        tmp_path.resolve()
    )
    old.rename(renamed)
    monkeypatch.setenv("SLURM_JOB_ID", "fake-cpu")
    monkeypatch.setenv("PHASE_RECOVERY_SOURCE", str(renamed.resolve()))
    assert runner.main(remote) == 2  # No results; source identity is valid, no task is submitted.
    with zipfile.ZipFile(remote.with_name(remote.name[:-6] + "_output") / "diagnostics.zip") as z:
        report = json.loads(z.read("recovery_status.json"))
    assert report["status"] == "incomplete" and len(report["missing_task_ids"]) == 15
    assert report["science_submissions"] == 0

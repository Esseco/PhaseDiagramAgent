"""Portable parallel training, scheduler dependencies and failed-task recovery."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import pytest


@pytest.fixture
def training_package(tmp_path):
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
    result = prepare_remote_finetune(
        state,
        {
            "mlip_finetune": settings,
            "upload_batches_directory": str(tmp_path),
            "mlip": {"version": "m1", "model_path": "/remote/base.model"},
            "python_environments": {"remote_mlip": "mace"},
        },
    )
    assert result["status"] == "awaiting_remote_training" and not result["submitted"]
    return next(iter(result["state"]["remote_finetune_jobs"].values()))


def test_new_package_has_independent_array_tasks_and_dependent_collection(training_package):
    job = training_package
    inputs = Path(job["directory"]) / "inputs"
    plan = json.loads((inputs / "training_plan.json").read_text(encoding="utf-8"))
    assert plan["execution"]["training_models"] == [
        "com_1",
        "com_2",
        "com_3",
        "com_4",
        "main_cv_1",
        "main_cv_2",
        "main_cv_3",
        "main_cv_4",
        "main_cv_5",
    ]
    assert plan["main_model"]["final_model"] == "com_1"
    assert job["handoff"]["job_count"] == 10
    assert job["handoff"]["max_parallel_training_jobs"] == 4
    with zipfile.ZipFile(job["handoff"]["upload_archive"]) as archive:
        submission = json.loads(archive.read("submission_plan.json"))
        assert submission["scripts"] == ["inputs/GPU.sh", "inputs/collect_results.sh"]
        assert submission["afterany"] == {"inputs/collect_results.sh": ["inputs/GPU.sh"]}
        worker = archive.read("inputs/GPU.sh").decode()
        assert "#SBATCH --array=0-8%4" in worker and "#SBATCH --gres=gpu:1" in worker
        assert "train-%A_%a.out" in worker and "train-%A_%a.err" in worker
        assert "conda activate mace" in worker
        assert "PHASE_HANDOFF_ROOT" not in worker
        assert "collect_training_results.py" not in archive.read("inputs/run_training.sh").decode()
        assert "collect_training_results.py" in archive.read("inputs/collect_results.sh").decode()
        assert "PHASE_HANDOFF_ROOT" in archive.read("inputs/collect_results.sh").decode()
        assert "inputs/run_training_task.py" in archive.namelist()


def test_parallel_package_rebuild_preserves_dependencies_and_configuration(training_package):
    from phase_agent.tools.remote.write_training_submission import prepare_training_handoff

    job = training_package
    directory = Path(job["directory"])
    inputs = directory / "inputs"
    original = (inputs / "com_1/mace_train.yaml").read_bytes()
    Path(job["handoff"]["upload_archive"]).unlink()
    rebuilt = prepare_training_handoff(directory, inputs)
    assert rebuilt["job_count"] == 10
    with zipfile.ZipFile(rebuilt["upload_archive"]) as archive:
        assert json.loads(archive.read("submission_plan.json"))["afterany"] == {
            "inputs/collect_results.sh": ["inputs/GPU.sh"]
        }
    assert (inputs / "com_1/mace_train.yaml").read_bytes() == original


def test_legacy_serial_inputs_are_not_rewritten(tmp_path):
    from phase_agent.tools.remote.write_training_submission import prepare_training_handoff

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    script = inputs / "GPU.sh"
    original = b"#!/bin/bash\n#SBATCH --gres=gpu:1\ntrue\n"
    script.write_bytes(original)
    (inputs / "training_plan.json").write_text("{}")
    handoff = prepare_training_handoff(tmp_path, inputs)
    assert handoff["job_count"] == 1 and script.read_bytes() == original
    with zipfile.ZipFile(handoff["upload_archive"]) as archive:
        assert json.loads(archive.read("submission_plan.json")) == {
            "schema": 1,
            "scripts": ["inputs/GPU.sh"],
        }


@pytest.mark.parametrize(
    "dependencies",
    [
        {"a.sh": ["missing.sh"]},
        {"a.sh": ["b.sh"], "b.sh": ["a.sh"]},
        {"missing.sh": ["a.sh"]},
        {"b.sh": "a.sh"},
    ],
)
def test_invalid_dependencies_are_rejected_before_dispatch(dependencies):
    from phase_agent.tools.remote.submission_dispatcher import submission_order

    with pytest.raises(ValueError):
        submission_order(["a.sh", "b.sh"], dependencies)


@pytest.mark.parametrize("resume", [False, True])
def test_emitted_dispatcher_waits_for_entire_array_and_does_not_resubmit(
    training_package, monkeypatch, resume
):
    import hashlib

    root = Path(training_package["handoff"]["folder"])
    spec = importlib.util.spec_from_file_location(
        "parallel_dispatch_test", root / "dispatch_tasks.py"
    )
    dispatch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dispatch)
    monkeypatch.setenv("SLURM_JOB_ID", "fake-controller")
    monkeypatch.setitem(
        sys.modules, "fcntl", SimpleNamespace(flock=lambda *_: None, LOCK_EX=1, LOCK_NB=2)
    )
    monkeypatch.setattr(dispatch.shutil, "which", lambda _: "fake-sbatch")
    monkeypatch.setattr(dispatch, "preflight_environment", lambda *_a, **_k: None)
    calls = []

    def submit(command, **kwargs):
        calls.append((command, kwargs["cwd"]))
        return SimpleNamespace(
            returncode=0, stdout=f"{100 + len(calls) + int(resume)}\n", stderr=""
        )

    monkeypatch.setattr(dispatch.subprocess, "run", submit)
    if resume:
        (root / ".submitted_jobs.json").write_text(
            json.dumps(
                {
                    "archive_sha256": hashlib.sha256(
                        (root / "inputs.zip").read_bytes()
                    ).hexdigest(),
                    "jobs": {"inputs/GPU.sh": {"status": "submitted", "job_id": "101"}},
                }
            ),
            encoding="utf-8",
        )
    dispatch.run(root)
    assert len(calls) == (1 if resume else 2)
    assert calls[-1][0] == [
        "sbatch",
        "--parsable",
        "--dependency=afterany:101",
        "collect_results.sh",
    ]
    assert all(cwd == root / "inputs" for _, cwd in calls)
    receipt = json.loads((root / ".submitted_jobs.json").read_text())
    assert receipt["jobs"]["inputs/GPU.sh"]["job_id"] == "101"
    dispatch.run(root)
    assert len(calls) == (1 if resume else 2)


@pytest.mark.parametrize("exit_code", [0, 2])
def test_one_array_worker_trains_only_its_model_and_records_exit(tmp_path, monkeypatch, exit_code):
    from phase_agent.tools.remote import run_training_task as worker

    names = ["com_1", "main_cv_1"]
    (tmp_path / "training_plan.json").write_text(
        json.dumps({"execution": {"training_models": names}})
    )
    for name in names:
        (tmp_path / name).mkdir()
        (tmp_path / name / "mace_train.yaml").write_text("seed: 123\n")
    calls = []

    def train(command, **kwargs):
        calls.append((command, kwargs["cwd"]))
        return SimpleNamespace(returncode=exit_code)

    monkeypatch.setattr(worker.subprocess, "run", train)
    assert worker.train(tmp_path, 1) == exit_code
    assert calls == [
        (
            [sys.executable, "-m", "mace.cli.run_train", "--config", "mace_train.yaml"],
            tmp_path / "main_cv_1",
        )
    ]
    marker = json.loads((tmp_path / "main_cv_1/training_task_status.json").read_text())
    assert marker["status"] == ("completed" if exit_code == 0 else "failed")
    assert marker["exit_code"] == exit_code
    assert not (tmp_path / "com_1/training_task_status.json").exists()
    with pytest.raises(ValueError):
        worker.train(tmp_path, -1)
    assert len(calls) == 1


def test_collection_rejects_failed_training_even_if_a_partial_model_exists(tmp_path, monkeypatch):
    from phase_agent.tools.remote import collect_training_results as collector
    from phase_agent.tools.local.recover_remote_training import inspect_training_results

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    names = ["com_1", "com_2", "main_cv_1"]
    plan = {
        "original_model_version": "m1",
        "execution": {"mode": "slurm_array", "training_models": names},
    }
    (inputs / "training_plan.json").write_text(json.dumps(plan))
    for name in names:
        member = inputs / name
        member.mkdir()
        (member / f"{name}.model").write_bytes(b"partial or complete model")
        (member / "training_task_status.json").write_text(
            json.dumps(
                {
                    "model_id": name,
                    "status": "failed" if name == "com_2" else "completed",
                    "exit_code": 1 if name == "com_2" else 0,
                }
            )
        )
    monkeypatch.setattr(collector, "__file__", str(inputs / "collect_training_results.py"))
    with pytest.raises(RuntimeError, match="com_2"):
        collector.main()
    marker = json.loads((tmp_path / "results/training.finished.json").read_text())
    assert marker["status"] == "failed" and marker["activated"] is False
    assert [row["model_id"] for row in marker["failed_training_tasks"]] == ["com_2"]
    report = inspect_training_results({"directory": str(tmp_path), "original_model_version": "m1"})
    assert report["status"] == "invalid_results" and "com_2" in " ".join(report["issues"])


def test_successful_parallel_training_collects_the_existing_recovery_format(
    training_package, monkeypatch
):
    from ase.calculators.calculator import Calculator, all_changes
    from phase_agent.tools.remote import collect_training_results as collector
    from phase_agent.tools.local.recover_remote_training import inspect_training_results

    job = training_package
    inputs = Path(job["directory"]) / "inputs"
    plan = job["report"]
    labels = plan["config"]["labels"]
    for name in plan["execution"]["training_models"]:
        member = inputs / name
        (member / f"{name}.model").write_bytes(b"synthetic model, never loaded")
        (member / "training_task_status.json").write_text(
            json.dumps(
                {
                    "model_id": name,
                    "status": "completed",
                    "exit_code": 0,
                }
            )
        )

    class FakeCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self, **_kwargs):
            super().__init__()

        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results = {
                "energy": atoms.info[labels.get("energy_key", "REF_energy")],
                "forces": atoms.arrays[labels.get("forces_key", "REF_forces")].copy(),
            }

    monkeypatch.setitem(sys.modules, "mace", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "mace.calculators", SimpleNamespace(MACECalculator=FakeCalculator)
    )
    monkeypatch.setattr(collector, "__file__", str(inputs / "collect_training_results.py"))
    collector.main()
    marker = json.loads((inputs.parent / "results/training.finished.json").read_text())
    assert marker["status"] == "completed" and marker["folds"] == 5
    assert marker["committee_count"] == 4 and marker["activated"] is False
    report = inspect_training_results(job)
    # Windows local fixture paths correctly require remote metadata, but the
    # complete metrics, model membership and result format must validate.
    assert report["status"] == "metadata_required"
    assert len(report["models"]) == 4
    assert float(report["kfold_metrics"]["energy_RMSE_meV_per_atom"]) == 0

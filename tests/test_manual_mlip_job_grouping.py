"""Manual upload has one Slurm submission per compatible MLIP batch."""

import json
import pytest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from phase_agent.tools.remote.manual_upload_runner import ManualUploadBatchRunner
from phase_agent.tools.remote.run_mlip_batch import run_mlip_batch


def _prepared_task(source):
    return {**source, "worker_job": {"structure_path": source["source_path"]}}


def _make_tasks(tmp_path, count, stage):
    structure = tmp_path / "source.vasp"
    structure.write_text("mock", encoding="utf-8")
    tasks = [
        {
            "task_id": f"T-{index}",
            "task_key": f"K-{index}",
            "stage": stage,
            "status": "pending",
            "branch_id": f"B-{index // 3}",
            "model_version": "mh-1",
            "source_path": str(structure),
        }
        for index in range(count)
    ]
    reservations = {row["task_key"]: {"status": "reserved"} for row in tasks}
    return {"tasks": tasks, "budget_reservations": reservations}


def test_relax_101_tasks_use_two_jobs(tmp_path):
    runner = ManualUploadBatchRunner(
        tmp_path / "upload",
        worker_command=["python3", "--executor", "x:y"],
        task_preparer=_prepared_task,
    )
    state = _make_tasks(tmp_path, 101, "relax_and_feature")
    first = runner.prepare(state)
    second = runner.prepare(first["state"])
    assert len(first["batch"]["task_ids"]) <= 100
    assert len(first["batch"]["task_ids"]) + len(second["batch"]["task_ids"]) == 101
    assert set(first["batch"]["task_ids"]).isdisjoint(second["batch"]["task_ids"])
    first_directory = Path(first["batch"]["upload_directory"])
    second_directory = Path(second["batch"]["upload_directory"])
    assert first_directory.relative_to(tmp_path / "upload").parts[0] == "epoch0"
    assert first["batch"]["search_group_index"] == 1
    assert first_directory.parent.parent == Path(first["batch"]["operation_directory"])
    assert first_directory.parent.parent.name.endswith("Relax-0001")
    assert first_directory.name == "Relax-submission-0001_remote-000001"
    assert first_directory.parent == second_directory.parent
    assert second_directory.name == "Relax-submission-0002_remote-000002"
    assert first["batch"]["mlip_round"] == 1
    assert first["batch"]["submission_index"] == 1
    assert (Path(first["batch"]["upload_directory"]) / "GPU.sh").is_file()
    script = (Path(first["batch"]["upload_directory"]) / "GPU.sh").read_bytes()
    assert script.startswith(b"#!/bin/bash\n") and b"\r" not in script
    assert b"#SBATCH --job-name=relax-remote-000001" in script
    assert b"job-name=test" not in script
    assert not list(Path(first["batch"]["upload_directory"]).glob("*/GPU.sh"))
    manifest = json.loads((Path(first["batch"]["upload_directory"]) / "manifest.json").read_text())
    assert all("\\" not in row["input_path"] for row in manifest)


def test_mc_21_tasks_use_three_jobs_without_splitting_branches(tmp_path):
    runner = ManualUploadBatchRunner(
        tmp_path / "upload",
        worker_command=["python3", "--executor", "x:y"],
        task_preparer=_prepared_task,
        stage_batch_sizes={"deep_search": 10},
    )
    state = _make_tasks(tmp_path, 21, "deep_search")
    first = runner.prepare(state)
    second = runner.prepare(first["state"])
    third = runner.prepare(second["state"])
    assert [len(row["batch"]["task_ids"]) for row in (first, second, third)] == [9, 9, 3]
    assert first["batch"]["search_group_index"] == 1
    assert Path(first["batch"]["upload_directory"]).parent.parent.name.endswith(
        "Relax-0001_MC-round-0001"
    )
    assert Path(second["batch"]["upload_directory"]).name == "MC-sampling-0002_remote-000002"
    assert Path(third["batch"]["upload_directory"]).name == "MC-sampling-0003_remote-000003"
    script = Path(first["batch"]["upload_directory"], "GPU.sh").read_text(encoding="utf-8")
    assert "#SBATCH --job-name=mc-remote-000001" in script


def test_two_mc_allocations_have_separate_results_and_never_mix(tmp_path):
    runner = ManualUploadBatchRunner(
        tmp_path / "upload",
        worker_command=["python3", "--executor", "x:y"],
        task_preparer=_prepared_task,
    )
    state = _make_tasks(tmp_path, 2, "deep_search")
    state["tasks"][0].update(upload_operation_id="first", segment_index=0)
    state["tasks"][1].update(upload_operation_id="second", segment_index=1)
    first = runner.prepare(state)
    second = runner.prepare(first["state"])
    assert first["batch"]["task_ids"] == ["T-0"]
    assert second["batch"]["task_ids"] == ["T-1"]
    assert first["batch"]["results_directory"] != second["batch"]["results_directory"]
    assert Path(second["batch"]["upload_directory"]).parent.parent.name.endswith(
        "Relax-0001_MC-round-0002"
    )


def test_new_model_version_starts_a_new_mlip_round(tmp_path):
    runner = ManualUploadBatchRunner(
        tmp_path / "upload",
        worker_command=["python3", "--executor", "x:y"],
        task_preparer=_prepared_task,
    )
    first_state = _make_tasks(tmp_path, 1, "deep_search")
    first_state["tasks"][0]["model_version"] = "mh-1"
    first = runner.prepare(first_state)
    second_task = _make_tasks(tmp_path, 1, "deep_search")
    second_task["tasks"][0].update(
        {"task_id": "T-new", "task_key": "K-new", "model_version": "mh-2"}
    )
    state = first["state"]
    state["tasks"].extend(second_task["tasks"])
    state["budget_reservations"]["K-new"] = {"status": "reserved"}
    second = runner.prepare(state)
    assert second["batch"]["mlip_round"] == 2
    assert second["batch"]["submission_index"] == 1
    assert "epoch1" in Path(second["batch"]["upload_directory"]).parts


def test_batch_continues_after_one_failed_task(tmp_path):
    root = tmp_path / "batch"
    root.mkdir()
    manifest = []
    for index, exit_code in enumerate((1, 0)):
        folder = root / f"{index:05d}-T-{index}"
        folder.mkdir()
        (folder / "run_mlip_task.py").write_text(
            "from pathlib import Path\n"
            "Path('task.finished.json').write_text('\\\"done\\\"')\n"
            f"raise SystemExit({exit_code})\n",
            encoding="utf-8",
        )
        manifest.append({"task_id": f"T-{index}", "input_path": f"{folder.name}/task.json"})
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert run_mlip_batch(root / "manifest.json") == ["T-0"]
    assert (root / "00001-T-1" / "task.finished.json").is_file()


def test_batch_runner_accepts_legacy_windows_manifest_paths(tmp_path):
    root = tmp_path / "legacy"
    task_dir = root / "00000-RELAX-old"
    task_dir.mkdir(parents=True)
    (task_dir / "task.json").write_text("{}", encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps([{"task_id": "RELAX-old", "input_path": "00000-RELAX-old\\task.json"}]),
        encoding="utf-8",
    )
    observed = {}

    def fake_run(command, *, cwd, **kwargs):
        observed["cwd"] = Path(cwd)
        (Path(cwd) / "task.finished.json").write_text("{}", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    with patch("phase_agent.tools.remote.run_mlip_batch.subprocess.run", side_effect=fake_run):
        failures = run_mlip_batch(root / "manifest.json", executor="fake.module:execute")
    assert failures == []
    assert observed["cwd"] == task_dir


@pytest.mark.parametrize("threshold", [None, 0.1, 0.2])
def test_mlip_single_point_batches_up_to_500_without_mixing_groups(tmp_path, threshold):
    runner = ManualUploadBatchRunner(tmp_path / "upload", worker_command=["worker"])
    state = _make_tasks(tmp_path, 501, "relax_and_feature")
    for task in state["tasks"]:
        task["parameters"] = {"model_refresh_operation": "predict", "refresh_force_threshold": threshold}
        task["search_group_index"] = 1
    selected = runner._select(state, state["tasks"])
    assert len(selected) == 500
    state["tasks"][1]["search_group_index"] = 2
    assert len(runner._select(state, state["tasks"])) == 500
    assert state["tasks"][1] not in runner._select(state, state["tasks"])
    runner.batch_sizes["mlip_single_point"] = 25
    assert len(runner._select(state, state["tasks"])) == 25


def test_relax_and_dft_batch_limits_do_not_change(tmp_path):
    runner = ManualUploadBatchRunner(tmp_path / "upload", worker_command=["worker"])
    relax = _make_tasks(tmp_path, 101, "relax_and_feature")
    assert len(runner._select(relax, relax["tasks"])) == 100
    dft = _make_tasks(tmp_path, 3, "dft_single_point")
    assert len(runner._select(dft, dft["tasks"])) == 1


def test_slurm_single_point_capacity_respects_explicit_global_limit(tmp_path):
    from phase_agent.tools.slurm.slurm_batch_runner import SlurmBatchRunner

    state = _make_tasks(tmp_path, 501, "relax_and_feature")
    for task in state["tasks"]:
        task["parameters"] = {"model_refresh_operation": "predict"}
    runner = SlurmBatchRunner(tmp_path / "slurm", worker_command=["worker"])
    assert len(runner._select_tasks(state, state["tasks"])) == 500
    limited = SlurmBatchRunner(tmp_path / "limited", worker_command=["worker"], max_batch_tasks=50)
    assert len(limited._select_tasks(state, state["tasks"])) == 50


@pytest.mark.parametrize("threshold", [None, 0.1, 0.2])
def test_predict_batch_reuses_calculator_and_keeps_individual_failures(tmp_path, monkeypatch, threshold):
    from phase_agent.science.mlip import model_interface
    from phase_agent.tools.remote import run_mlip_batch as batch_module
    import phase_agent.tools.remote.run_single_task as single_task

    root = tmp_path / "inputs" / "batch"
    root.mkdir(parents=True)
    entries = []
    for i in range(3):
        directory = root / str(i)
        directory.mkdir()
        task = {
            "task_id": str(i),
            "task_key": str(i),
            "stage": "relax_and_feature",
            "parameters": {"model_refresh_operation": "predict", "refresh_force_threshold": threshold},
            "worker_job": {"operation": "predict", "model_path": "/model"},
        }
        (directory / "task.json").write_text(json.dumps(task), encoding="utf-8")
        entries.append({"task_id": str(i), "input_path": f"{i}/task.json"})
    (root / "manifest.json").write_text(json.dumps(entries), encoding="utf-8")
    loads, calls = [], []

    def create(options, *, model_path=None):
        loads.append(model_path)
        return object()

    def execute(task):
        calculator = model_interface.create_calculator({"device": "cuda"}, model_path="/model")
        calls.append(calculator)
        if task["task_id"] == "1":
            raise ValueError("bad frame")
        return {"status": "completed"}

    monkeypatch.setattr(model_interface, "create_calculator", create)
    monkeypatch.setattr(single_task, "_load", lambda reference: execute)
    monkeypatch.setattr(batch_module, "export_batch_result", lambda *a, **k: root)
    monkeypatch.setattr("phase_agent.tools.remote.transfer_bundle.result_bundle", lambda *a: None)
    assert batch_module.run_mlip_batch(root / "manifest.json", executor="fake:execute") == ["1"]
    assert loads == ["/model"]
    assert len(calls) == 3 and calls[0] is calls[1] is calls[2]
    assert json.loads((root / "1" / "result.json").read_text())["status"] == "failed"
    assert model_interface.create_calculator is create


def test_direct_slurm_prediction_reuses_model_and_records_each_result(tmp_path, monkeypatch):
    from phase_agent.science.mlip import model_interface
    from phase_agent.tools.remote.run_mlip_batch import run_slurm_predict_batch
    from phase_agent.tools.slurm import run_slurm_array_task as slurm_module

    manifest = []
    for i in range(3):
        directory = tmp_path / str(i)
        directory.mkdir()
        task = {"task_id": str(i), "task_key": str(i)}
        task_path = directory / "task.json"
        task_path.write_text(json.dumps(task), encoding="utf-8")
        manifest.append(
            {
                "array_index": i,
                "task_id": str(i),
                "task_key": str(i),
                "input_path": str(task_path),
                "result_path": str(directory / "result.json"),
            }
        )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    loads, calculators = [], []

    def factory(options, *, model_path=None):
        loads.append(model_path)
        return object()

    def execute(task):
        calculators.append(
            model_interface.create_calculator({"device": "cuda"}, model_path="model")
        )
        return {"status": "failed" if task["task_id"] == "1" else "completed"}

    monkeypatch.setattr(model_interface, "create_calculator", factory)
    monkeypatch.setattr(slurm_module, "load_executor", lambda ref: execute)
    assert run_slurm_predict_batch(path, "fake:executor") == ["1"]
    assert loads == ["model"] and len(set(map(id, calculators))) == 1
    assert all((tmp_path / str(i) / "task.finished.json").is_file() for i in range(3))
    assert model_interface.create_calculator is factory


def test_refresh_batch_shares_mace_calculator_with_inline_relax(tmp_path, monkeypatch):
    from Process_MLIP import calculators
    from phase_agent.science.mlip import model_interface
    from phase_agent.tools.remote.run_mlip_batch import _reuse_calculators

    model = tmp_path / "model"
    model.write_text("fixture", encoding="utf8")
    loaded = []

    def factory(model_path, *, device="cuda", head=None, default_dtype="float64",
                enable_cueq=False, enable_symmetrix=False, symmetrix_use_kokkos=False,
                symmetrix_species=None):
        loaded.append(model_path)
        return object()

    monkeypatch.setattr(calculators, "create_mace_calculator", factory)
    with _reuse_calculators():
        first = model_interface.create_calculator(
            {"framework": "mace", "device": "cuda"}, model_path=str(model)
        )
        relaxed = calculators.create_mace_calculator(
            model, device="cuda", default_dtype="float64"
        )
        assert first is relaxed
    assert len(loaded) == 1
    assert calculators.create_mace_calculator is factory


def test_slurm_predict_script_uses_batch_entry_only_for_predict(tmp_path):
    from phase_agent.tools.slurm.slurm_batch_runner import SlurmBatchRunner

    runner = SlurmBatchRunner(tmp_path, worker_command=["python3", "--executor", "pkg:execute"])
    predict = runner._script(
        "id", tmp_path / "manifest.json", 2, "relax_and_feature", single_point=True
    )
    relax = runner._script("id", tmp_path / "manifest.json", 2, "relax_and_feature")
    assert "--slurm-predict" in predict and "pkg:execute" in predict
    assert "--slurm-predict" not in relax and "--array-index" in relax


def test_real_predict_executor_reuses_ase_calculator_across_frames(tmp_path, monkeypatch):
    from ase import Atoms
    from ase.calculators.emt import EMT
    from ase.io import write
    from phase_agent.science.mlip import model_interface
    from phase_agent.tools.remote import run_mlip_batch as batch_module

    root = tmp_path / "inputs" / "batch"
    root.mkdir(parents=True)
    model = tmp_path / "model"
    model.write_text("fake model", encoding="utf-8")
    entries = []
    for i, distance in enumerate((2.5, 2.8)):
        folder = root / str(i)
        folder.mkdir()
        write(
            folder / "initial.vasp",
            Atoms("Cu2", positions=[[0, 0, 0], [distance, 0, 0]], cell=[10, 10, 10], pbc=True),
            format="vasp",
        )
        task = {
            "task_id": str(i),
            "task_key": str(i),
            "structure_id": str(i),
            "stage": "relax_and_feature",
            "parameters": {"model_refresh_operation": "predict"},
            "worker_job": {
                "operation": "predict",
                "structure_path": "initial.vasp",
                "model_path": str(model),
                "parameters": {"model_refresh_operation": "predict"},
            },
        }
        (folder / "task.json").write_text(json.dumps(task), encoding="utf-8")
        entries.append({"task_id": str(i), "input_path": f"{i}/task.json"})
    (root / "manifest.json").write_text(json.dumps(entries), encoding="utf-8")
    loads = []

    def calculator(options, *, model_path=None):
        loads.append(model_path)
        return EMT()

    monkeypatch.setattr(model_interface, "create_calculator", calculator)
    monkeypatch.setattr(batch_module, "export_batch_result", lambda *a, **k: root)
    monkeypatch.setattr("phase_agent.tools.remote.transfer_bundle.result_bundle", lambda *a: None)
    assert (
        batch_module.run_mlip_batch(
            root / "manifest.json",
            executor="phase_agent.science.mlip.slurm_executor:execute_mlip_task",
        )
        == []
    )
    assert loads == [str(model)]
    results = [json.loads((root / str(i) / "result.json").read_text()) for i in range(2)]
    assert all(r["status"] == "completed" for r in results)
    assert results[0]["outputs"]["energy"] != results[1]["outputs"]["energy"]

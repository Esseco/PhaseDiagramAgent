"""Manual upload has one Slurm submission per compatible MLIP batch."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from execution_layer.remote.manual_upload_runner import ManualUploadBatchRunner
from execution_layer.remote.run_mlip_batch import run_mlip_batch


def _prepared_task(source):
    return {**source, "worker_job": {"structure_path": source["source_path"]}}


def _make_tasks(tmp_path, count, stage):
    structure = tmp_path / "source.vasp"
    structure.write_text("mock", encoding="utf-8")
    tasks = [{"task_id": f"T-{index}", "task_key": f"K-{index}", "stage": stage,
              "status": "pending", "branch_id": f"B-{index // 3}",
              "model_version": "mh-1", "source_path": str(structure)}
             for index in range(count)]
    reservations = {row["task_key"]: {"status": "reserved"} for row in tasks}
    return {"tasks": tasks, "budget_reservations": reservations}


def test_relax_101_tasks_use_two_jobs(tmp_path):
    runner = ManualUploadBatchRunner(
        tmp_path / "upload", worker_command=["python3", "--executor", "x:y"],
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
    assert first_directory.parts[-3:] == (
        "MLIP-round-0001_mh-1", "Relax-screening",
        "Relax-submission-0001_remote-000001")
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
        tmp_path / "upload", worker_command=["python3", "--executor", "x:y"],
        task_preparer=_prepared_task,
    )
    state = _make_tasks(tmp_path, 21, "deep_search")
    first = runner.prepare(state)
    second = runner.prepare(first["state"])
    third = runner.prepare(second["state"])
    assert [len(row["batch"]["task_ids"]) for row in (first, second, third)] == [9, 9, 3]
    assert Path(first["batch"]["upload_directory"]).parts[-2:] == (
        "MC-search", "MC-sampling-0001_remote-000001")
    assert Path(second["batch"]["upload_directory"]).name == "MC-sampling-0002_remote-000002"
    assert Path(third["batch"]["upload_directory"]).name == "MC-sampling-0003_remote-000003"
    script = Path(first["batch"]["upload_directory"], "GPU.sh").read_text(encoding="utf-8")
    assert "#SBATCH --job-name=mc-remote-000001" in script


def test_new_model_version_starts_a_new_mlip_round(tmp_path):
    runner = ManualUploadBatchRunner(
        tmp_path / "upload", worker_command=["python3", "--executor", "x:y"],
        task_preparer=_prepared_task,
    )
    first_state = _make_tasks(tmp_path, 1, "deep_search")
    first_state["tasks"][0]["model_version"] = "mh-1"
    first = runner.prepare(first_state)
    second_task = _make_tasks(tmp_path, 1, "deep_search")
    second_task["tasks"][0].update({"task_id": "T-new", "task_key": "K-new",
                                     "model_version": "mh-2"})
    state = first["state"]
    state["tasks"].extend(second_task["tasks"])
    state["budget_reservations"]["K-new"] = {"status": "reserved"}
    second = runner.prepare(state)
    assert second["batch"]["mlip_round"] == 2
    assert second["batch"]["submission_index"] == 1
    assert "MLIP-round-0002_mh-2" in Path(second["batch"]["upload_directory"]).parts


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
            f"raise SystemExit({exit_code})\n", encoding="utf-8")
        manifest.append({"task_id": f"T-{index}",
                         "input_path": f"{folder.name}/task.json"})
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert run_mlip_batch(root / "manifest.json") == ["T-0"]
    assert (root / "00001-T-1" / "task.finished.json").is_file()


def test_batch_runner_accepts_legacy_windows_manifest_paths(tmp_path):
    root = tmp_path / "legacy"
    task_dir = root / "00000-RELAX-old"
    task_dir.mkdir(parents=True)
    (task_dir / "task.json").write_text("{}", encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps([{
        "task_id": "RELAX-old", "input_path": "00000-RELAX-old\\task.json"
    }]), encoding="utf-8")
    observed = {}

    def fake_run(command, *, cwd, **kwargs):
        observed["cwd"] = Path(cwd)
        (Path(cwd) / "task.finished.json").write_text("{}", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    with patch("execution_layer.remote.run_mlip_batch.subprocess.run", side_effect=fake_run):
        failures = run_mlip_batch(root / "manifest.json", executor="fake.module:execute")
    assert failures == []
    assert observed["cwd"] == task_dir

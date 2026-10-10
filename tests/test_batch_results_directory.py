"""A downloadable batch results folder remains tied to immutable task inputs."""

import json
from pathlib import Path
from unittest.mock import patch

from phase_agent.tools.remote.batch_runner import RemoteBatchRunner
from phase_agent.tools.remote.export_batch_result import export_batch_result
from phase_agent.tools.remote.finalize_vasp import finalize_vasp
from phase_agent.tools.remote.integrity import file_checksum
from phase_agent.tools.slurm.run_slurm_array_task import run_slurm_array_task


def test_relax_result_is_collected_from_stage_results_folder(tmp_path):
    stage = tmp_path / "stage"
    batch = stage / "batch"
    task_dir = batch / "00000-RELAX-A"
    task_dir.mkdir(parents=True)
    task = {"task_id": "RELAX-A", "task_key": "relax-a", "batch_id": "remote-000001",
            "task_checksum": "input-a", "protocol_version": 1, "input_file_version": "task-json-v1"}
    (task_dir / "task.json").write_text(json.dumps(task), encoding="utf-8")
    final = task_dir / "final.vasp"
    final.write_text("final structure", encoding="utf-8")
    result = {**task, "stage": "relax_and_feature", "status": "completed",
              "outputs": {"energy": -12.5, "structure_path": str(final),
                          "structure_checksum": file_checksum(final)}}
    result_path = task_dir / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    marker = {**task, "status": "completed", "result_checksum": file_checksum(result_path)}
    (task_dir / "task.finished.json").write_text(json.dumps(marker), encoding="utf-8")

    exported = export_batch_result(task_dir, stage / "results")
    assert sorted(path.name for path in exported.iterdir()) == ["final.vasp", "result.json", "task.finished.json"]
    state = {"tasks": [{**task, "stage": "relax_and_feature", "status": "pending",
                       "input_path": str(task_dir / "task.json"),
                       "result_path": str(exported / "result.json")}],
             "slurm_batches": [{"batch_id": task["batch_id"], "upload_directory": str(batch),
                                "results_directory": str(stage / "results")}]}
    collected = RemoteBatchRunner(tmp_path / "unused", worker_command=[]).collect_results_with_report(state)
    assert collected["report"]["recovered_count"] == 1
    assert collected["results"][0]["outputs"]["structure_path"] == str((exported / "final.vasp").resolve())


def test_failed_dft_finalization_exports_checked_result(tmp_path):
    stage = tmp_path / "stage"
    task_dir = stage / "batch" / "00000-DFT-A"
    task_dir.mkdir(parents=True)
    task = {"task_id": "DFT-A", "task_key": "dft-a", "batch_id": "remote-000001",
            "task_checksum": "input-dft", "protocol_version": 1,
            "input_file_version": "task-json-v1", "stage": "dft_single_point"}
    (task_dir / "task.json").write_text(json.dumps(task), encoding="utf-8")
    result = finalize_vasp(task_dir, exit_code=9)
    exported = stage / "results" / task_dir.name
    assert result["status"] == "failed"
    assert (exported / "result.json").is_file()
    assert json.loads((exported / "task.finished.json").read_text(encoding="utf-8"))["result_checksum"] == file_checksum(exported / "result.json")
    state = {"tasks": [{**task, "status": "pending", "input_path": str(task_dir / "task.json"),
                       "result_path": str(exported / "result.json")}],
             "slurm_batches": [{"batch_id": task["batch_id"], "upload_directory": str(task_dir.parent)}]}
    collected = RemoteBatchRunner(tmp_path / "unused", worker_command=[]).collect_results_with_report(state)
    assert collected["report"]["recovered_count"] == 1


def test_completed_dft_result_relinks_downloaded_contcar(tmp_path):
    stage = tmp_path / "stage"
    task_dir = stage / "batch" / "00000-DFT-B"
    task_dir.mkdir(parents=True)
    task = {"task_id": "DFT-B", "task_key": "dft-b", "batch_id": "remote-000002",
            "task_checksum": "input-b", "protocol_version": 1,
            "input_file_version": "task-json-v1", "stage": "dft_relax"}
    (task_dir / "task.json").write_text(json.dumps(task), encoding="utf-8")
    (task_dir / "CONTCAR").write_text("final DFT structure", encoding="utf-8")
    parsed = {**task, "status": "completed", "outputs": {"energy": -20.0,
              "energy_unit": "eV", "structure_path": str(task_dir / "CONTCAR")}}
    with patch("phase_agent.tools.remote.finalize_vasp.parse_vasp_result", return_value=parsed):
        finalize_vasp(task_dir)
    exported = stage / "results" / task_dir.name
    state = {"tasks": [{**task, "status": "pending", "input_path": str(task_dir / "task.json"),
                       "result_path": str(exported / "result.json")}],
             "slurm_batches": [{"batch_id": task["batch_id"], "upload_directory": str(task_dir.parent)}]}
    collected = RemoteBatchRunner(tmp_path / "unused", worker_command=[]).collect_results_with_report(state)
    assert collected["report"]["recovered_count"] == 1
    assert collected["results"][0]["outputs"]["structure_path"] == str((exported / "CONTCAR").resolve())


def test_array_worker_calculates_in_input_folder_and_exports_result(tmp_path):
    stage = tmp_path / "stage"
    batch = stage / "batch"
    task_dir = batch / "00000-MC-A"
    task_dir.mkdir(parents=True)
    (task_dir / "initial.vasp").write_text("input", encoding="utf-8")
    task = {"task_id": "MC-A", "task_key": "mc-a", "batch_id": "remote-000003",
            "task_checksum": "input-mc", "calculation_directory": task_dir.name}
    (task_dir / "task.json").write_text(json.dumps(task), encoding="utf-8")
    manifest = [{"array_index": 0, **task, "input_path": f"{task_dir.name}/task.json",
                 "result_path": f"../results/{task_dir.name}/result.json"}]
    manifest_path = batch / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def executor(job):
        working_directory = Path(job["result_path"]).parent
        assert working_directory == task_dir
        final = working_directory / "final.vasp"
        final.write_text("MC result", encoding="utf-8")
        return {"status": "completed", "outputs": {"structure_path": str(final),
                "structure_checksum": file_checksum(final)}}

    assert run_slurm_array_task(manifest_path, 0, executor=executor)["status"] == "completed"
    assert (stage / "results" / task_dir.name / "final.vasp").is_file()
    assert (stage / "results" / task_dir.name / "task.finished.json").is_file()


def test_numbered_dft_results_and_visible_training_file(tmp_path):
    from types import SimpleNamespace
    from pymatgen.core import Structure, Lattice
    stage = tmp_path / "DFT-single-point"
    submission = stage / "DFT-single-point-submission-0006_remote-000029"
    task_dir = submission / "00000-DFT-B"
    task_dir.mkdir(parents=True)
    task = {"task_id": "DFT-B", "task_key": "dft-b", "structure_id": "S1",
            "batch_id": "remote-000029", "task_checksum": "input-b", "model_version": "m1",
            "protocol_version": 1, "input_file_version": "task-json-v1", "stage": "dft_single_point"}
    (task_dir / "task.json").write_text(json.dumps(task))
    (task_dir / "vasprun.xml").write_text("stub")
    structure = Structure(Lattice.cubic(4), ["Na", "O"], [[0, 0, 0], [.5, .5, .5]])
    parsed = SimpleNamespace(converged=True, final_energy=-8., vasp_version="6",
        ionic_steps=[{"structure": structure, "e_0_energy": -8., "forces": [[0, 0, 0]]*2}])
    with patch("pymatgen.io.vasp.outputs.Vasprun", return_value=parsed):
        finalize_vasp(task_dir)
    exported = stage / "results" / submission.name
    assert not (exported / "CONTCAR").exists()
    assert not (exported / "final_structure.vasp").exists()
    training = json.loads((exported / "training.json").read_text())
    assert len(training) == 1 and training[0]["energy"] == -8.
    assert training[0]["data_id"] == "DFT-B:frame:0"
    result = json.loads((exported / "result.json").read_text())
    assert result["outputs"]["training_frame_count"] == 1
    assert result["outputs"]["training_checksum"] == file_checksum(exported / "training.json")
    state = {"tasks": [{**task, "status": "pending", "input_path": str(task_dir / "task.json"),
                       "result_path": str(stage / "results" / task_dir.name / "result.json")}],
             "slurm_batches": [{"batch_id": task["batch_id"], "upload_directory": str(submission)}]}
    runner = RemoteBatchRunner(tmp_path / "unused", worker_command=[])
    assert runner.collect_results_with_report(state)["report"]["recovered_count"] == 1
    (exported / "training.json").write_text("tampered")
    bad = runner.collect_results_with_report(state)
    assert bad["report"]["invalid_count"] == 1
    assert bad["report"]["recovered_count"] == 0

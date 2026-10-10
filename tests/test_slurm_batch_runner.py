import json

import pytest

from phase_agent.configuration.defaults.default_slurm_cluster_config import default_slurm_cluster_config
from phase_agent.tools.slurm.run_slurm_array_task import run_slurm_array_task
from phase_agent.tools.slurm.slurm_batch_runner import SlurmBatchRunner
from phase_agent.science.dft.create_atomate_workflow import generate_dft_workflow_with_atomate
from phase_agent.science.dft.parse_vasp_result import write_vasp_result


def _state():
    tasks = [
        {"task_id": "T1", "task_key": "K1", "stage": "deep_search", "status": "pending"},
        {"task_id": "T2", "task_key": "K2", "stage": "deep_search", "status": "pending"},
        {"task_id": "T3", "task_key": "K3", "stage": "deep_search", "status": "pending"},
    ]
    return {
        "tasks": tasks,
        "pending_tasks": list(tasks),
        "budget_reservations": {
            "K1": {"status": "reserved", "reserved_cost": 2.0},
            "K2": {"status": "reserved", "reserved_cost": 3.0},
            "K3": {"status": "released", "reserved_cost": 1.0},
        },
    }


def test_runner_batches_only_reserved_tasks_within_batch_budget(tmp_path):
    runner = SlurmBatchRunner(
        tmp_path / "slurm", worker_command=["python", "worker.py"], max_batch_cost=2.5
    )
    result = runner.prepare(_state())

    assert result["status"] == "prepared"
    assert result["batch"]["task_ids"] == ["T1"]
    assert result["batch"]["planned_cost"] == 2.0
    manifest = json.loads((tmp_path / "slurm/slurm-000001/manifest.json").read_text())
    assert manifest[0]["task_key"] == "K1"
    assert "SLURM_ARRAY_TASK_ID" in (tmp_path / "slurm/slurm-000001/submit.sbatch").read_text()

    repeated = runner.prepare(result["state"])
    assert repeated["status"] == "no_tasks"


def test_mlip_profile_is_rendered_for_mc_and_relax(tmp_path):
    runner = SlurmBatchRunner(
        tmp_path / "slurm", worker_command=["python3", "worker.py"],
        stage_profiles=default_slurm_cluster_config(),
    )
    result = runner.prepare(_state())
    script = (tmp_path / "slurm/slurm-000001/submit.sbatch").read_text()
    assert "#SBATCH --partition=v100m3" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH --ntasks=2" in script
    assert "conda activate mace" in script
    assert 'export CUDA_HOME="${cuda_path}"' in script
    assert result["batch"]["backend"] == "slurm"


def test_dft_batch_requires_atomate_dispatcher(tmp_path):
    state = _state()
    state["tasks"] = state["pending_tasks"] = [{
        "task_id": "T1", "task_key": "K1", "stage": "dft_single_point", "status": "pending"
    }]
    runner = SlurmBatchRunner(tmp_path / "missing", worker_command=["worker"])
    with pytest.raises(ValueError, match="atomate dispatcher"):
        runner.prepare(state)

    runner = SlurmBatchRunner(
        tmp_path / "atomate", worker_command=["worker"],
        stage_profiles=default_slurm_cluster_config(),
        dispatcher=lambda task: {
            "status": "pending", "outputs": {"generator": "atomate",
            "workflow_path": str(tmp_path / "workflow.json")},
        },
    )
    result = runner.prepare(state)
    payload = json.loads(
        (tmp_path / "atomate/slurm-000001/00000-T1/task.json").read_text()
    )
    assert payload["atomate"]["generator"] == "atomate"
    assert result["batch"]["task_ids"] == ["T1"]


def test_collect_terminal_results_for_main_loop(tmp_path):
    runner = SlurmBatchRunner(tmp_path / "slurm", worker_command=["worker"])
    prepared = runner.prepare(_state())["state"]
    result_path = tmp_path / "slurm/slurm-000001/00000-T1/result.json"
    result_path.write_text(json.dumps({"status": "completed", "actual_cost": 1.5}))
    (result_path.parent / "task.finished.json").write_text(json.dumps({"status": "completed"}))

    recovered = runner.collect_results(prepared)
    assert recovered == [{
        "status": "completed", "actual_cost": 1.5, "task_id": "T1", "task_key": "K1"
    }]


def test_array_worker_always_writes_terminal_result(tmp_path):
    runner = SlurmBatchRunner(tmp_path / "slurm", worker_command=["worker"])
    prepared = runner.prepare(_state())
    manifest = prepared["batch"]["manifest_path"]
    completed = run_slurm_array_task(
        manifest, 0, executor=lambda task: {"status": "completed", "actual_cost": 1.25}
    )
    assert completed["task_id"] == "T1"
    assert json.loads((tmp_path / "slurm/slurm-000001/00000-T1/result.json").read_text())["actual_cost"] == 1.25

    failed = run_slurm_array_task(
        manifest, 1, executor=lambda task: (_ for _ in ()).throw(RuntimeError("backend down"))
    )
    assert failed["status"] == "failed"
    assert "backend down" in failed["error"]


def test_dft_batch_uses_atomate_inputs_and_direct_vasp_slurm(tmp_path):
    workflow_path = tmp_path / "workflow.json"
    workflow_path.write_text(json.dumps({"name": "wf"}))
    state = _state()
    state["tasks"] = state["pending_tasks"] = [{
        "task_id": "T1", "task_key": "K1", "stage": "dft_single_point", "status": "pending"
    }]
    runner = SlurmBatchRunner(
        tmp_path / "atomate", worker_command=["unused"], submit=False,
        dispatcher=lambda task: {"status": "pending", "outputs": {
            "generator": "atomate", "workflow_path": str(workflow_path)}},
        stage_profiles=default_slurm_cluster_config(),
    )
    prepared = runner.prepare(state)
    script = (tmp_path / "atomate/slurm-000001/submit.sbatch").read_text()
    assert prepared["batch"]["backend"] == "atomate_vasp"
    assert "module load nvhpc-hpcx fftw/3.3.10-nvhpc" in script
    assert "mpirun -np ${SLURM_NPROCS} vasp_std" in script
    assert "phase_agent.science.dft.parse_vasp_result" in script
    assert "LaunchPad" not in script


def test_atomate_workflow_materializes_vasp_inputs(tmp_path):
    class Workflow:
        name = "static"
        def to_dict(self): return {"name": self.name}

    def writer(workflow, directory):
        paths = []
        for name in ("INCAR", "POSCAR", "KPOINTS", "POTCAR"):
            path = directory / name; path.write_text(name); paths.append(str(path))
        return paths

    result = generate_dft_workflow_with_atomate(
        "structure", calculation_type="singlepoint", work_directory=tmp_path / "dft",
        workflow_factory=lambda structure, **kwargs: Workflow(), input_writer=writer,
    )
    assert result["status"] == "pending"
    assert len(result["input_files"]) == 4


def test_vasp_parser_writes_recoverable_result(tmp_path):
    (tmp_path / "task.json").write_text(json.dumps({
        "task_id": "D1", "task_key": "K1", "structure_id": "S1",
        "stage": "dft_single_point", "result_path": str(tmp_path / "result.json"),
    }))
    result = write_vasp_result(tmp_path, "result.json", exit_code=9)
    assert result["status"] == "failed"
    assert "code 9" in json.loads((tmp_path / "result.json").read_text())["error"]

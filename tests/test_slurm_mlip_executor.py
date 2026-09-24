import json
from unittest.mock import patch

from config_layer.defaults.default_slurm_cluster_config import default_slurm_cluster_config
from data_layer.ledger.phase_data_manager import PhaseDataManager
from execution_layer.slurm.slurm_batch_runner import SlurmBatchRunner
from scientific_layer.mlip.slurm_executor import create_mlip_task_preparer, execute_mlip_task


H = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_mlip_mc_task_is_self_contained_and_uses_incremental_budget(tmp_path):
    structure_path = tmp_path / "input.vasp"; structure_path.write_text("mock")
    manager = PhaseDataManager({"P": ["O3"], "H": {"O3": [H]}, "TM_ratio": {"Fe": 1}})
    branch = manager.add_branch(P="O3", H=H, x=1, T=["Fe"], composition={"Fe": 1})
    structure = manager.add_structure(branch_id=branch, arrangement={"a": 1}, source_path=structure_path)
    task = {
        "task_id": "MC1", "task_key": "K1", "branch_id": branch,
        "stage": "deep_search", "status": "pending", "incremental_budget": 20,
        "planned_relative_cost": 2.0,
    }
    state = {
        "tasks": [task], "pending_tasks": [task],
        "budget_reservations": {"K1": {"status": "reserved", "reserved_cost": 2.0}},
    }
    runner = SlurmBatchRunner(
        tmp_path / "batches",
        worker_command=["python3", "-m", "execution_layer.slurm.run_slurm_array_task",
                        "--executor", "scientific_layer.mlip.slurm_executor:execute_mlip_task"],
        stage_profiles=default_slurm_cluster_config(),
        task_preparer=create_mlip_task_preparer(
            manager, {}, {"mlip": {"model_path": "/models/mace.model", "version": "m1"}}
        ),
    )
    prepared = runner.prepare(state)
    payload = json.loads((tmp_path / "batches/slurm-000001/00000-MC1/task.json").read_text())
    assert payload["structure_id"] == structure
    assert payload["worker_job"]["operation"] == "mc"
    assert payload["worker_job"]["segment_budget"] == 20
    assert payload["worker_job"]["model_version"] == "m1"
    assert "slurm_array_task" in (tmp_path / "batches/slurm-000001/submit.sbatch").read_text()

    with patch("scientific_layer.mlip.slurm_executor.run_mace_worker", return_value={
        "status": "completed", "energy": -3.0, "energy_unit": "eV",
        "structure_path": str(structure_path), "checkpoint": "checkpoint.json",
    }):
        result = execute_mlip_task(payload)
    assert result["status"] == "completed"
    assert result["actual_cost"] is None  # an estimate is not a measured cost
    assert result["outputs"]["mlip_version"] == "m1"


def test_mlip_relax_uses_same_gpu_worker(tmp_path):
    structure_path = tmp_path / "input.vasp"; structure_path.write_text("mock")
    manager = PhaseDataManager({"P": ["O3"], "H": {"O3": [H]}, "TM_ratio": {"Fe": 1}})
    branch = manager.add_branch(P="O3", H=H, x=1, T=["Fe"], composition={"Fe": 1})
    structure = manager.add_structure(branch_id=branch, arrangement={"a": 1}, source_path=structure_path)
    preparer = create_mlip_task_preparer(manager, {}, {"mlip": {"model_path": "m.model"}})
    prepared = preparer({
        "task_id": "R1", "task_key": "RK", "structure_id": structure,
        "stage": "relax_and_feature", "status": "pending",
        "calculation_directory": str(tmp_path / "relax"), "result_path": str(tmp_path / "relax/result.json"),
    })
    assert prepared["worker_job"]["operation"] == "relax"
    assert prepared["worker_job"]["segment_budget"] is None


def test_committee_uses_first_member_as_main_without_python_path_setting(tmp_path):
    structure_path = tmp_path / "input.vasp"; structure_path.write_text("mock")
    manager = PhaseDataManager({"P": ["O3"], "H": {"O3": [H]}, "TM_ratio": {"Fe": 1}})
    branch = manager.add_branch(P="O3", H=H, x=1, T=["Fe"], composition={"Fe": 1})
    structure = manager.add_structure(branch_id=branch, arrangement={"a": 1}, source_path=structure_path)
    paths = [f"/models/com_{index}.model" for index in range(1, 5)]
    prepared = create_mlip_task_preparer(manager, {}, {"mlip": {
        "model_path": "/models/mace-mh-1.model", "model_paths": paths,
        "main_model_index": 0,
    }})({"task_id": "R2", "task_key": "RK2", "structure_id": structure,
         "stage": "relax_and_feature", "status": "pending",
         "calculation_directory": str(tmp_path / "relax2"),
         "result_path": str(tmp_path / "relax2/result.json")})
    job = prepared["worker_job"]
    assert job["model_path"] is None
    assert job["model_paths"] == paths
    assert job["parameters"]["main_model_index"] == 0
    assert job["parameters"]["save_relax_traj"] is False
    assert "backend_python_paths" not in job

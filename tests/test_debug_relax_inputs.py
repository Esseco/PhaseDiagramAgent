"""Debug mode prepares portable Relax files without running MACE or submitting."""

import json
from pathlib import Path
from types import SimpleNamespace

from phase_agent.configuration.defaults.default_budget_rules import default_budget_rules
from phase_agent.decisions.agent.choose_debug_next_action import choose_debug_next_action
from phase_agent.tools.local.prepare_relax_upload_batches import prepare_relax_upload_batches
from phase_agent.tools.remote.manual_upload_runner import ManualUploadBatchRunner
from phase_agent.tools.workflows.run_tool_step import _prepare_debug_relax_screen_action


def test_debug_relax_preparation_is_portable_and_idempotent(tmp_path):
    structure = tmp_path / "initial.vasp"
    structure.write_text("mock structure; preparation must not parse or relax it", encoding="utf-8")
    manager = SimpleNamespace(data={
        "branches": {"B-1": {"branch_id": "B-1", "P": "O3", "structure_ids": ["S-1"]}},
        "structures": {"S-1": {"branch_id": "B-1", "source_path": str(structure),
            "composition": {"Na": 1, "Fe": 1, "Mn": 1, "O": 4},
            "metadata": {"initialization_method": "electrostatic_top10_random3_layer_occupied"}}},
    }, boundary={"P": ["O3"]})
    config = {"config_version": "config-test", "budgets": default_budget_rules(),
              "mlip": {"name": "mace-mh-1", "model_path": "/remote/mace-mh-1.model"},
              "bohb": {"relax_structures_per_branch": 3},
              "upload_batches_directory": str(tmp_path / "upload"),
              "supercomputer": {"worker": {"command": ["python3", "-m",
                  "phase_agent.tools.slurm.run_slurm_array_task", "--executor",
                  "phase_agent.science.mlip.slurm_executor:execute_mlip_task"]},
                  "batch_sizes": {"relax_and_feature": 100}}}
    state = {"dedup_gate": {"status": "ready"}, "tasks": []}
    action = choose_debug_next_action(state, manager, config,
                                      allowed_tools=["prepare_local_batch_files"], user_message="继续")
    assert action["tool"] == "prepare_local_batch_files"
    result = prepare_relax_upload_batches(action=action, context={
        "effective_config": config, "manager": manager, "event_state": state,
        "config_version": "config-test", "phase_references": {},
    })
    assert result["status"] == "prepared" and result["task_count"] == 1
    batch = Path(result["batches"][0]["directory"])
    task_file = next(batch.glob("*/task.json"))
    task = json.loads(task_file.read_text(encoding="utf-8"))
    assert task["worker_job"]["structure_path"] == "initial.vasp"
    assert (task_file.parent / "initial.vasp").is_file()
    script = (batch / "GPU.sh").read_text(encoding="utf-8")
    assert "conda activate mace" in script
    assert "run_mlip_batch" in script
    assert not (task_file.parent / "GPU.sh").exists()
    assert "--executor" in (batch / "run_mlip_batch.py").read_text(encoding="utf-8")
    assert not list(batch.glob("*/run_mlip_task.py"))
    assert not (batch / "submit.sbatch").exists()
    assert not list(batch.glob("*/result.json"))
    again = prepare_relax_upload_batches(action=action, context={
        "effective_config": config, "manager": manager, "event_state": result["state"],
        "config_version": "config-test", "phase_references": {},
    })
    assert again["status"] == "already_prepared" and again["task_count"] == 0
    assert len(list((tmp_path / "upload").rglob("*remote-*"))) == 1


def test_relax_preparation_includes_all_registered_branches_in_one_compatible_job(tmp_path):
    structures = {}
    branches = {}
    for branch_id, count in (("B-1", 2), ("B-2", 1)):
        ids = []
        for index in range(count):
            sid = f"S-{branch_id}-{index}"
            source = tmp_path / f"{sid}.vasp"
            source.write_text("mock structure", encoding="utf-8")
            structures[sid] = {"branch_id": branch_id, "source_path": str(source),
                "composition": {"Na": 1, "Fe": 1, "Mn": 1, "O": 4},
                "metadata": {"initialization_method": "electrostatic_top10_random3_layer_occupied"}}
            ids.append(sid)
        branches[branch_id] = {"branch_id": branch_id, "P": "O3", "structure_ids": ids}
    manager = SimpleNamespace(data={"branches": branches, "structures": structures},
                              boundary={"P": ["O3"]})
    config = {"config_version": "v1", "budgets": default_budget_rules(),
              "mlip": {"name": "mace-mh-1", "model_path": "/remote/model"},
              "bohb": {"relax_structures_per_branch": 3},
              "upload_batches_directory": str(tmp_path / "upload"),
              "supercomputer": {"worker": {"command": ["python3", "-m", "worker", "--executor",
                  "phase_agent.science.mlip.slurm_executor:execute_mlip_task"]}}}
    state = {"dedup_gate": {"status": "ready"}, "tasks": []}
    action = choose_debug_next_action(state, manager, config,
                                      allowed_tools=["prepare_local_batch_files"])
    assert set(action["target_ids"]) == {"B-1", "B-2"}
    revised = _prepare_debug_relax_screen_action(
        {"tool": "run_calculation_stage", "stage": "relax_screen", "target_ids": ["B-1"]},
        state, {"manager": manager, "effective_config": config}, config,
        ["prepare_local_batch_files"], invocation_id="review-1",
    )
    assert set(revised["target_ids"]) == {"B-1", "B-2"}
    result = prepare_relax_upload_batches(action={**action, "target_ids": ["B-1"]}, context={
        "effective_config": config, "manager": manager, "event_state": state,
        "config_version": "v1", "phase_references": {},
    })
    assert result["task_count"] == 3 and result["batch_count"] == 1
    grouped = sorted(len(batch["task_ids"]) for batch in result["batches"])
    assert grouped == [3]
    plan = json.loads((tmp_path / "upload" / "RELAX_UPLOAD_PLAN.json").read_text(encoding="utf-8"))
    assert plan["branch_count"] == 2 and plan["task_count"] == 3
    assert sorted(len(branch["task_ids"]) for branch in plan["branches"]) == [1, 2]
    again = prepare_relax_upload_batches(action=action, context={
        "effective_config": config, "manager": manager, "event_state": result["state"],
        "config_version": "v1", "phase_references": {},
    })
    assert again["status"] == "already_prepared" and len(again["state"]["tasks"]) == 3


def test_manual_mc_task_gets_own_structure_and_script(tmp_path):
    structure = tmp_path / "source.vasp"
    structure.write_text("mock", encoding="utf-8")
    full_na = tmp_path / "full_na.vasp"
    full_na.write_text("mock full Na", encoding="utf-8")
    task = {"task_id": "MC-1", "task_key": "mc-key", "stage": "deep_search",
            "status": "pending", "model_version": "mh-1"}
    runner = ManualUploadBatchRunner(
        tmp_path / "upload", worker_command=["python3", "-m", "worker", "--executor",
                                            "phase_agent.science.mlip.slurm_executor:execute_mlip_task"],
        task_preparer=lambda row: {**row, "worker_job": {"structure_path": str(structure),
            "operation": "mc", "model_path": "/remote/mh-1.model",
            "parameters": {"full_na_structure": str(full_na)}}},
    )
    result = runner.prepare({"tasks": [task], "budget_reservations": {"mc-key": {"status": "reserved"}}})
    directory = next(Path(result["batch"]["upload_directory"]).glob("*/task.json")).parent
    assert (directory / "initial.vasp").is_file()
    assert (directory / "full_na_structure.vasp").is_file()
    payload = json.loads((directory / "task.json").read_text(encoding="utf-8"))
    assert payload["worker_job"]["structure_path"] == "initial.vasp"
    assert payload["worker_job"]["parameters"]["full_na_structure"] == "full_na_structure.vasp"
    assert "conda activate mace" in (directory.parent / "GPU.sh").read_text(encoding="utf-8")
    assert "run_mlip_batch" in (directory.parent / "GPU.sh").read_text(encoding="utf-8")
    assert (directory.parent / "run_mlip_batch.py").is_file()
    assert not (directory / "run_mlip_task.py").exists()
    assert not (directory / "GPU.sh").exists()
    assert not (directory.parent / "submit.sbatch").exists()


def test_manual_dft_task_uses_atomate_inputs_and_vasp_script(tmp_path):
    def atomate(task):
        directory = Path(task["work_directory"])
        for name in ("POSCAR", "INCAR", "KPOINTS", "POTCAR"):
            (directory / name).write_text("mock", encoding="utf-8")
        return {"outputs": {"generator": "atomate"}}

    task = {"task_id": "DFT task 1", "task_key": "dft-key", "stage": "dft_single_point",
            "status": "pending"}
    runner = ManualUploadBatchRunner(tmp_path / "upload", worker_command=[], dispatcher=atomate)
    result = runner.prepare({"tasks": [task], "budget_reservations": {"dft-key": {"status": "reserved"}}})
    directory = next(Path(result["batch"]["upload_directory"]).glob("*/task.json")).parent
    script = (directory / "GPU.sh").read_text(encoding="utf-8")
    assert "vasp_std" in script
    assert "#SBATCH --job-name=dft-sp-DFT-task-1" in script
    assert "job-name=test" not in script
    assert not (directory / "run_mlip_task.py").exists()

import json
from types import SimpleNamespace
from unittest.mock import patch

from decision_layer.agent.choose_debug_next_action import choose_debug_next_action
from execution_layer.local.prepare_mc_upload_batches import prepare_mc_upload_batches
from execution_layer.remote.resolve_local_relax_structure import resolve_local_relax_structure
from scientific_layer.mlip.slurm_executor import execute_mlip_task


def test_normal_relax_stop_is_eligible_for_phase_analysis(tmp_path):
    structure = tmp_path / "final.vasp"
    structure.write_text("mock", encoding="utf-8")
    with patch("scientific_layer.mlip.slurm_executor.run_mace_worker", return_value={
        "status": "completed", "relax_stopped_normally": True, "structure_path": str(structure),
        "energy": -5.0, "energy_unit": "eV"}):
        result = execute_mlip_task({"stage": "relax_and_feature", "structure_id": "S1",
            "result_path": str(tmp_path / "result.json"), "worker_job": {
                "operation": "relax", "structure_path": str(structure), "model_path": "mh1",
                "model_version": "mh1", "parameters": {}}})
    assert result["converged"] is True
    assert result["outputs"]["energy"] == -5.0


def test_final_structure_must_be_downloaded_before_recovery(tmp_path):
    result = {"status": "completed", "outputs": {"structure_path": "/remote/job/pool/rank_0.vasp"}}
    assert resolve_local_relax_structure(result, tmp_path) is None
    final = tmp_path / "pool" / "rank_0.vasp"
    final.parent.mkdir()
    final.write_text("mock")
    resolved = resolve_local_relax_structure(result, tmp_path)
    assert resolved["outputs"]["structure_path"] == str(final.resolve())
    bad = {"status": "completed", "outputs": {"structure_path": "/remote/job/pool/rank_0.vasp",
           "structure_checksum": "incorrect"}}
    assert resolve_local_relax_structure(bad, tmp_path) is None


def test_mc_inputs_use_relaxed_structure_and_full_na_template(tmp_path):
    relaxed = tmp_path / "relaxed.vasp"; relaxed.write_text("relaxed")
    full_na = tmp_path / "full_na.vasp"; full_na.write_text("full Na")
    manager = SimpleNamespace(data={"branches": {"B1": {"structure_ids": ["S1"]}},
        "structures": {"S1": {"branch_id": "B1", "source_path": str(tmp_path / "initial.vasp"),
                              "full_na_structure_path": str(full_na)}}}, boundary={"P": ["O3"]})
    tasks = [{"task_id": f"MC{i}", "task_key": f"mc:{i}", "stage": "deep_search",
              "status": "pending", "branch_id": "B1", "structure_id": "S1",
              "structure_path": str(relaxed), "model_version": "mh1", "incremental_budget": 10,
              "parameters": {"max_mc_steps": 10, "patience_steps": 3, "min_improvement": 0.001}}
             for i in range(21)]
    state = {"dedup_gate": {"status": "ready", "valid_structure_ids": ["S1"]},
             "tasks": tasks, "budget_reservations": {
                 row["task_key"]: {"status": "reserved"} for row in tasks}}
    config = {"upload_batches_directory": str(tmp_path / "upload"),
              "mlip": {"model_path": "/remote/mh1.model", "version": "mh1"},
              "supercomputer": {"worker": {"command": ["python3", "--executor",
                                "scientific_layer.mlip.slurm_executor:execute_mlip_task"]}}}
    next_action = choose_debug_next_action(state, manager, config,
        allowed_tools=["prepare_local_batch_files"], user_message="继续")
    assert next_action["parameters"]["mode"] == "mc_inputs"
    result = prepare_mc_upload_batches(action=next_action, context={"event_state": state,
        "effective_config": config, "manager": manager, "phase_references": {}, "config_version": "v1"})
    assert result["status"] == "prepared" and result["batch_count"] == 2
    assert sorted(len(batch["task_ids"]) for batch in result["batches"]) == [1, 20]
    first = tmp_path / "upload" / result["batches"][0]["batch_id"]
    task = json.loads(next(first.glob("*/task.json")).read_text(encoding="utf-8"))
    assert (first / task["calculation_directory"] / "initial.vasp").read_text() == "relaxed"
    assert (first / task["calculation_directory"] / "full_na_structure.vasp").read_text() == "full Na"
    assert task["worker_job"]["segment_budget"] == 10

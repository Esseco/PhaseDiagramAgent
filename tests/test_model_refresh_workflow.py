from copy import deepcopy
import json
from types import SimpleNamespace

from ase import Atoms
from ase.io import write

from config_layer.defaults.default_budget_rules import default_budget_rules
from data_layer.models.require_structure_refresh import require_structure_refresh
from execution_layer.local.prepare_local_batch_files import prepare_local_batch_files
from execution_layer.workflows.model_refresh_gate import run_model_refresh_gate, advance_refresh
from execution_layer.workflows.run_tool_step import run_tool_step


def setup(tmp_path):
    path = tmp_path / "original.vasp"
    write(path, Atoms("NaFeO2", positions=[[0,0,0], [1,1,1], [2,2,2], [3,3,3]], cell=[5,5,5], pbc=True), format="vasp")
    manager = SimpleNamespace(data={"structures": {"s": {"branch_id": "b", "source_path": str(path),
        "composition": {"Na": 1, "Fe": 1, "O": 2}}}, "branches": {"b": {"structure_ids": ["s"]}}}, boundary={})
    config = {"budgets": default_budget_rules(), "mlip": {"version": "old", "model_path": "/models/old.model"},
        "upload_batches_directory": str(tmp_path / "submissions"),
        "supercomputer": {"worker": {"command": ["python", "-m", "execution_layer.remote.run_slurm_array_task", "--executor", "scientific_layer.mlip.slurm_executor:execute_mlip_task"]}}}
    state = {"tasks": [], "active_model": {"version": "new", "model_path": "/models/new.model"},
        "active_model_version": "new", "dedup_gate": {"status": "ready", "valid_structure_ids": ["s"]},
        "phase_diagrams": {"mlip": {"status": "completed", "model_version": "old", "entries": [
            {"structure_id": "s", "structure_path": str(path), "ehull": .005, "ehull_unit": "eV/atom", "phase": "O3"}]}}}
    require_structure_refresh(state, "old", "new")
    session = {"status": "confirmed", "confirmed_snapshot": {"config_version": "c", "config": {
        **config, "agent": {"allowed_tools": ["prepare_local_batch_files"]}}}}
    context = {"manager": manager, "effective_config": config, "config_version": "c"}
    registry = {"prepare_local_batch_files": {"handler": prepare_local_batch_files}}
    return state, session, context, registry


def test_refresh_requires_own_approval_and_generates_portable_single_point(tmp_path):
    state, session, context, registry = setup(tmp_path)
    first = run_model_refresh_gate(state, session, registry=registry, context=context,
                                   human_feedback={"decision": "approve", "comment": "同意"})
    assert first["status"] == "awaiting_approval"
    assert not (tmp_path / "submissions").exists()
    second = run_model_refresh_gate(first["state"], session, registry=registry, context=context,
                                    human_feedback={"decision": "approve", "comment": "同意"})
    assert second["status"] == "prepared", second
    task = second["state"]["tasks"][0]
    portable = json.loads(__import__("pathlib").Path(task["input_path"]).read_text(encoding="utf-8"))
    assert portable["worker_job"]["operation"] == "predict"
    assert portable["worker_job"]["model_path"] == "/models/new.model"
    assert portable["worker_job"]["structure_path"] == "initial.vasp"
    assert "structure_path" not in portable
    assert "Model-refresh-0001" in task["input_path"] and "Single-point" in task["input_path"]
    assert "epoch1_new" in task["input_path"]
    assert second["state"]["model_refresh"]["wave"] == 0


def recovered_state(tmp_path):
    state, session, context, registry = setup(tmp_path)
    first = run_model_refresh_gate(state, session, registry=registry, context=context)
    prepared = run_model_refresh_gate(first["state"], session, registry=registry, context=context,
                                      human_feedback="同意")["state"]
    task = prepared["tasks"][0]
    task.update(status="completed", converged=False, outputs={"single_point_completed": True,
        "structure_path": context["manager"].data["structures"]["s"]["source_path"], "forces": [[.1,0,0]]})
    prepared["phase_records"] = [{"record_id": "r", "source_task_id": task["task_id"], "source_version": "new"}]
    prepared["phase_diagrams"]["mlip"] = {"status": "completed", "model_version": "new",
        "entries": [{"record_id": "r", "ehull": .002, "ehull_unit": "eV/atom"}]}
    return prepared, session, context, registry


def test_bounded_supplement_and_no_third_wave(tmp_path):
    state, session, context, registry = recovered_state(tmp_path)
    result = run_model_refresh_gate(state, session, registry=registry, context=context)
    assert result["status"] == "prepared", result
    assert result["state"]["model_refresh"]["wave"] == 1
    tasks = result["state"]["tasks"]
    assert len(tasks) == 2
    tasks[-1].update(status="completed", converged=True)
    result["state"]["phase_records"].append({"record_id": "r2", "source_task_id": tasks[-1]["task_id"], "source_version": "new"})
    result["state"]["phase_diagrams"]["mlip"]["entries"].append({"record_id": "r2", "ehull": 0, "ehull_unit": "eV/atom"})
    final, waiting = advance_refresh(result["state"])
    assert waiting is None
    assert final["model_refresh"]["status"] == "completed"
    assert len(final["tasks"]) == 2


def test_old_model_hull_cannot_supply_supplement_ehull(tmp_path):
    state, _, _, _ = recovered_state(tmp_path)
    state["phase_diagrams"]["mlip"]["model_version"] = "old"
    final, waiting = advance_refresh(state)
    assert waiting and "不会沿用旧模型能量" in waiting
    assert final["model_refresh"]["wave"] == 0


def test_public_tool_step_intercepts_search_before_llm(tmp_path):
    state, session, context, registry = setup(tmp_path)
    response = run_tool_step(state, session, registry=registry, context=context,
                             execution_mode="interactive", invocation_id="refresh-test")
    assert response["status"] == "awaiting_approval"
    assert "refresh-test" in response["state"]["pending_execution_policies"]
    assert response["agent_proposal"]["estimated_cost"]["estimated_total_cost"] > 0


def test_changed_input_requires_fresh_approval(tmp_path):
    state, session, context, registry = setup(tmp_path)
    first = run_model_refresh_gate(state, session, registry=registry, context=context)
    path = __import__("pathlib").Path(context["manager"].data["structures"]["s"]["source_path"])
    path.write_text(path.read_text()+"\n", encoding="utf-8")
    second = run_model_refresh_gate(first["state"], session, registry=registry, context=context, human_feedback="同意")
    assert second["status"] == "awaiting_approval"
    assert not (tmp_path / "submissions").exists()


def test_budget_never_silently_truncates(tmp_path):
    state, session, context, registry = setup(tmp_path)
    context["effective_config"]["budgets"]["total_relative_cost"] = .0001
    session["confirmed_snapshot"]["config"]["budgets"]["total_relative_cost"] = .0001
    first = run_model_refresh_gate(state, session, registry=registry, context=context)
    second = run_model_refresh_gate(first["state"], session, registry=registry, context=context, human_feedback="同意")
    assert second["status"] == "rejected"
    assert not second["state"]["tasks"]
    assert not (tmp_path / "submissions").exists()


def test_failure_waiver_does_not_change_task_status(tmp_path):
    state, _, _, _ = recovered_state(tmp_path)
    state["tasks"][0]["status"] = "failed"
    updated, waiting = advance_refresh(state)
    assert waiting and updated["model_refresh"]["status"] == "waiting_results"
    updated, _ = advance_refresh(state, "不再等待刷新")
    assert updated["tasks"][0]["status"] == "failed"
    assert updated["tasks"][0]["refresh_wait_waived"] is True


def test_refresh_single_point_enters_versioned_hull_without_faking_convergence(tmp_path):
    from data_layer.ledger.collect_calculation_results import _phase_record
    from analysis_layer.phase.update_phase_diagram import update_phase_diagram
    manager = SimpleNamespace(data={"structures": {"s": {"branch_id": "b"}}, "branches": {"b": {"P": "O3"}}})
    records = []
    for index, composition in enumerate(({"Fe": 1, "O": 2}, {"Na": 1, "Fe": 1, "O": 2})):
        result = {"status": "completed", "task_id": str(index), "converged": False, "model_version": "new",
            "model_refresh_id": "approved", "parameters": {"model_refresh_operation": "predict"},
            "outputs": {"energy": -10-index, "energy_unit": "eV", "mlip_version": "new",
                "composition": composition, "single_point_completed": True, "actual_phase": "O3",
                "phase_identification": {"status": "identified"}}}
        record = _phase_record(manager, "s", f"r{index}", "relax_and_feature", result)
        assert record is not None
        records.append(record)
        ordinary = deepcopy(result)
        ordinary.pop("model_refresh_id")
        assert _phase_record(manager, "s", "not-converged", "relax_and_feature", ordinary) is None
    records.append({**records[0], "record_id": "old-record", "source_version": "old", "model_version": "old", "energy": -1000})
    diagram = update_phase_diagram(records, active_model_version="new")["diagrams"]["mlip"]
    assert diagram["status"] == "completed"
    assert {r["source_version"] for r in diagram["entries"]} == {"new"}
    assert len(diagram["entries"]) == 2

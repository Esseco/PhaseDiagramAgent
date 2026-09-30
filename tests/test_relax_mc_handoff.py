import json
import hashlib
import pytest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from analysis_layer.phase.branch_relax_hull import build_relax_hull
from config_layer.defaults.default_layered_search_config import default_layered_search_config
from decision_layer.agent.choose_debug_next_action import choose_debug_next_action
from execution_layer.dispatch.create_tool_registry import create_tool_registry
from execution_layer.local.prepare_mc_upload_batches import prepare_mc_upload_batches
from execution_layer.local.prepare_relax_upload_batches import prepare_relax_upload_batches
from execution_layer.policy.execution_policy import build_agent_proposal
from execution_layer.remote.resolve_local_relax_structure import resolve_local_relax_structure
from execution_layer.remote.summarize_manual_upload_wait import summarize_manual_upload_wait
from execution_layer.workflows.create_active_learning_handlers import _allocate_mc_bohb
from execution_layer.workflows.run_tool_step import run_tool_step
from run.open_webui_api import format_workflow_reply
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


@pytest.mark.parametrize("second_segment", [False, True])
def test_mc_inputs_use_relaxed_structure_and_full_na_template(tmp_path, second_segment):
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
    if second_segment:
        for task in tasks:
            task.pop("structure_id")
            task["segment_index"] = 1
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
    assert result["status"] == "prepared" and result["batch_count"] == 3
    assert sorted(len(batch["task_ids"]) for batch in result["batches"]) == [1, 10, 10]
    first = Path(result["batches"][0]["directory"])
    task = json.loads(next(first.glob("*/task.json")).read_text(encoding="utf-8"))
    assert (first / task["calculation_directory"] / "initial.vasp").read_text() == "relaxed"
    assert (first / task["calculation_directory"] / "full_na_structure.vasp").read_text() == "full Na"
    assert task["worker_job"]["segment_budget"] == 10


def _completed_relax_fixture(tmp_path):
    initial = tmp_path / "initial.vasp"
    initial.write_text("initial", encoding="utf-8")
    full_na = tmp_path / "full_na.vasp"
    full_na.write_text("full Na", encoding="utf-8")
    signature = hashlib.sha256(b"{}").hexdigest()[:12]
    structures, tasks, rows = {}, [], []
    for index in range(3):
        sid = f"S{index}"
        final = tmp_path / f"relaxed-{index}.vasp"
        final.write_text(f"relaxed {index}", encoding="utf-8")
        structures[sid] = {"branch_id": "B1", "source_path": str(initial),
            "composition": {"Na": 1, "Fe": 1, "Mn": 1, "O": 4},
            "full_na_structure_path": str(full_na),
            "metadata": {"initialization_method": "electrostatic_top10_random3_layer_occupied"}}
        tasks.append({"task_id": f"RELAX-{index}",
            "task_key": f"relax-screen:m1:{signature}:{sid}",
            "branch_id": "B1", "structure_id": sid, "stage": "relax_and_feature",
            "status": "completed", "model_version": "m1", "converged": True,
            "outputs": {"relax_stopped_normally": True, "structure_path": str(final),
                "composition": {"Na": 1, "Fe": 1, "Mn": 1, "O": 4},
                "energy": -40.0 - index, "energy_unit": "eV"}})
        rows.append({"branch_id": "B1", "structure_id": sid,
            "structure_path": str(final), "composition": {"Na": 1, "Fe": 1, "Mn": 1, "O": 4},
            "energy": -40.0 - index, "energy_unit": "eV", "converged": True,
            "model_version": "m1"})
    branch = {"branch_id": "B1", "P": "O3", "H": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
              "x": "1/2", "T": ["Fe", "Mn"], "structure_ids": list(structures)}
    manager = SimpleNamespace(data={"branches": {"B1": branch}, "structures": structures},
                              boundary={"P": ["O3"]})
    pool = build_relax_hull(rows, model_version="m1")
    state = {"tasks": tasks, "dedup_gate": {"status": "ready"},
             "branch_hull_batches": {pool["version"]: pool},
             "current_branch_hull_version": pool["version"]}
    state["phase_diagrams"] = {"mlip": {
        "method": "mlip", "status": "completed", "model_version": "m1",
        "version": "test-phase-v1", "entries": [{
            "structure_id": row["structure_id"], "structure_path": row["structure_path"],
            "normalized_total_energy": row["energy"],
            "ehull": (row["energy"] + 42) / 6, "ehull_unit": "eV/atom",
            "phase_identification_status": "identified", "phase": "O3",
        } for row in rows]}}
    config = default_layered_search_config()
    config["mlip"].update({"name": "m1", "version": "m1", "model_path": "/remote/model"})
    config["mlip"].pop("relax_parameters")
    config["upload_batches_directory"] = str(tmp_path / "upload")
    config["round_strategy"]["rule_default"]["mc_budget"] = 100
    return manager, state, config


def test_completed_relax_advances_to_mc_and_prepares_upload_files(tmp_path):
    manager, state, config = _completed_relax_fixture(tmp_path)
    action = choose_debug_next_action(state, manager, config,
        allowed_tools=config["agent"]["allowed_tools"], user_message="继续")
    assert action["tool"] == "allocate_mc_bohb"
    assert action["target_ids"] == ["B1"]
    result = _allocate_mc_bohb(action=action, context={"manager": manager,
        "phase_references": {}, "effective_config": config, "event_state": state,
        "config_version": "v1", "execution_mode": "interactive"})
    assert result["status"] == "completed"
    assert result["mc_upload"]["status"] == "prepared"
    assert len(result["actions"]) == 1
    assert len([row for row in result["state"]["tasks"] if row["stage"] == "relax_and_feature"]) == 3
    batch = Path(result["mc_upload"]["batches"][0]["directory"])
    mc_dir = next(batch.glob("*/task.json")).parent
    assert (mc_dir / "initial.vasp").read_text(encoding="utf-8") == "relaxed 2"
    assert (mc_dir / "full_na_structure.vasp").read_text(encoding="utf-8") == "full Na"
    assert "#SBATCH --job-name=mc-" in (batch / "GPU.sh").read_text(encoding="utf-8")


def test_completed_first_mc_proposes_one_second_segment_only(tmp_path):
    manager, state, config = _completed_relax_fixture(tmp_path)
    final = tmp_path / "mc-final.vasp"
    final.write_text("mc final", encoding="utf-8")
    first = {"task_id": "MC-first", "task_key": "first-mc", "stage": "deep_search",
             "branch_id": "B1", "model_version": "m1", "status": "completed",
             "segment_index": 0, "tier_index": 0, "max_mc_steps": 10,
             "min_improvement": 0.001, "energy_improvement": 0.01,
             "stop_reason": "max_steps", "result_path": str(final),
             "outputs": {"energy_per_atom": -6.0}}
    state["tasks"].append(first)
    state["tiered_mc_state"] = {"segments": [{**first, "status": "pending"}],
                               "processed_task_keys": []}
    state["phase_diagrams"]["mlip"]["entries"].append({
        "structure_id": "MC-final", "structure_path": str(final),
        "normalized_total_energy": -42.0, "ehull": 0.0,
        "ehull_unit": "eV/atom", "phase": "O3",
        "phase_identification_status": "identified"})
    action = choose_debug_next_action(state, manager, config,
        allowed_tools=config["agent"]["allowed_tools"], user_message="继续")
    assert action["tool"] == "allocate_mc_bohb"
    assert action["parameters"]["round_kind"] == "second"
    assert action["parameters"]["budget_preview"]["selected_branch_count"] == 1
    assert action["parameters"]["budget_preview"]["allocations"][0]["segment_index"] == 1
    result = _allocate_mc_bohb(action=action, context={"manager": manager,
        "phase_references": {}, "effective_config": config, "event_state": state,
        "config_version": "v1", "execution_mode": "automatic"})
    assert result["status"] == "completed"
    assert len(result["actions"]) == 1
    again = _allocate_mc_bohb(action=action, context={"manager": manager,
        "phase_references": {}, "effective_config": config,
        "event_state": result["state"], "config_version": "v1", "execution_mode": "automatic"})
    assert again["reason"] == "second_mc_round_already_allocated"


def test_second_mc_requires_confirmed_policy_and_current_mc_phase_entry(tmp_path):
    manager, state, config = _completed_relax_fixture(tmp_path)
    final = tmp_path / "mc-final.vasp"
    final.write_text("mc final", encoding="utf-8")
    first = {"task_id": "MC-first", "task_key": "first-mc", "stage": "deep_search",
             "branch_id": "B1", "model_version": "m1", "status": "completed",
             "segment_index": 0, "tier_index": 0, "max_mc_steps": 10,
             "min_improvement": 0.001, "energy_improvement": 0.01,
             "result_path": str(final)}
    state["tasks"].append(first)
    state["tiered_mc_state"] = {"segments": [first]}
    assert choose_debug_next_action(state, manager, config,
        allowed_tools=config["agent"]["allowed_tools"], user_message="继续") is None
    state["phase_diagrams"]["mlip"]["entries"].append({
        "structure_id": "MC-final", "structure_path": str(final),
        "normalized_total_energy": -42.0, "ehull": 0.0,
        "ehull_unit": "eV/atom", "phase": "O3",
        "phase_identification_status": "identified"})
    config["mc_policy"]["second_segment_enabled"] = False
    assert choose_debug_next_action(state, manager, config,
        allowed_tools=config["agent"]["allowed_tools"], user_message="继续") is None


def test_mc_approval_stops_if_phase_diagram_changed(tmp_path):
    manager, state, config = _completed_relax_fixture(tmp_path)
    action = choose_debug_next_action(state, manager, config,
        allowed_tools=config["agent"]["allowed_tools"], user_message="继续")
    assert action["parameters"]["phase_diagram_version"] == "test-phase-v1"
    state["phase_diagrams"]["mlip"]["version"] = "new-phase-version"
    result = _allocate_mc_bohb(action=action, context={"manager": manager,
        "phase_references": {}, "effective_config": config, "event_state": state,
        "config_version": "v1", "execution_mode": "interactive"})
    assert result["status"] == "not_configured"
    assert result["reason"] == "approved_phase_diagram_changed"
    assert not [task for task in result["state"].get("tasks", [])
                if task["stage"] == "deep_search"]


def test_verified_legacy_relax_results_advance_to_mc_after_config_migration(tmp_path):
    manager, state, config = _completed_relax_fixture(tmp_path)
    old_version, new_version = "config-old", "config-new"
    config["config_version"] = new_version
    config["mlip"].update({"name": "mace-mh-1", "version": "mace-mh-1",
                           "mace_head": "omat_pbe", "relax_parameters": {
                               "fmax": 0.05, "relax_steps": 150, "relax_cell": True,
                               "mace_default_dtype": "float64"}})
    for task in state["tasks"]:
        task.update({"config_version": old_version, "model_version": "mace-mh-1"})
        task["task_key"] = f"relax-screen:mace-mh-1:{hashlib.sha256(b'{}').hexdigest()[:12]}:{task['structure_id']}"
        task["outputs"].update({"mace_head": "omat_pbe", "fmax_target_ev_per_angstrom": 0.05,
                                "cell_relaxed": True, "relax_steps_used": 150,
                                "relax_stopped_normally": True})
    state["confirmed_config_version"] = new_version
    state["config_migrations"] = [{
        "type": "approved_migration_with_verified_legacy_mace_defaults",
        "from": old_version,
        "to": new_version,
        "compatibility": {
            "basis": "verified_legacy_mace_mh_1_worker_defaults",
            "model": "mace-mh-1",
            "legacy_config_version": old_version,
            "relax_task_count": 3,
            "verified_result_fields": {
                "mace_head": "omat_pbe",
                "fmax_target_ev_per_angstrom": 0.05,
                "cell_relaxed": True,
                "relax_steps_used_max": 150,
            },
        },
    }]
    pool = state["branch_hull_batches"][state["current_branch_hull_version"]]
    pool["model_version"] = "mace-mh-1"
    state["phase_diagrams"]["mlip"]["model_version"] = "mace-mh-1"
    for row in pool["records"]:
        row["model_version"] = "mace-mh-1"

    action = choose_debug_next_action(state, manager, config,
        allowed_tools=config["agent"]["allowed_tools"], user_message="继续")

    assert action["tool"] == "allocate_mc_bohb"
    assert action["target_ids"] == ["B1"]


def test_scheduling_revision_keeps_relax_pool_and_does_not_block_mc(tmp_path):
    manager, state, config = _completed_relax_fixture(tmp_path)
    legacy, screened, current = "config-legacy", "config-screened", "config-current"
    config["mlip"].update({"name": "mace-mh-1", "version": "mace-mh-1",
                           "mace_head": "omat_pbe", "relax_parameters": {
                               "fmax": 0.05, "relax_steps": 150, "relax_cell": True,
                               "mace_default_dtype": "float64"}})
    for task in state["tasks"]:
        task.update({"config_version": legacy, "model_version": "mace-mh-1"})
        task["outputs"].update({"mace_head": "omat_pbe", "fmax_target_ev_per_angstrom": 0.05,
                                "cell_relaxed": True, "relax_steps_used": 150})
    state.update({"confirmed_config_version": current, "config_migrations": [
        {"type": "approved_migration_with_verified_legacy_mace_defaults",
         "from": legacy, "to": screened, "compatibility": {
             "basis": "verified_legacy_mace_mh_1_worker_defaults", "model": "mace-mh-1",
             "legacy_config_version": legacy, "verified_result_fields": {
                 "mace_head": "omat_pbe", "fmax_target_ev_per_angstrom": 0.05,
                 "cell_relaxed": True, "relax_steps_used_max": 150}}},
        {"type": "confirmed_generation_policy_revision", "from": screened, "to": current},
    ]})
    pool = state["branch_hull_batches"][state["current_branch_hull_version"]]
    pool["model_version"] = "mace-mh-1"
    state["phase_diagrams"]["mlip"]["model_version"] = "mace-mh-1"
    for row in pool["records"]:
        row["model_version"] = "mace-mh-1"

    # A newer supplementary branch has no Relax result. It must not block MC
    # for the already screened pool.
    extra = tmp_path / "extra.vasp"
    extra.write_text("extra", encoding="utf-8")
    manager.data["structures"]["S-new"] = {
        "branch_id": "B-new", "source_path": str(extra),
        "composition": {"Na": 1, "Fe": 1, "Mn": 1, "O": 4},
        "metadata": {"initialization_method": "electrostatic_top10_random3_layer_occupied"},
    }
    manager.data["branches"]["B-new"] = {
        "branch_id": "B-new", "P": "O3", "x": "1/2", "det_H": 2,
        "structure_ids": ["S-new"],
    }
    action = choose_debug_next_action(
        state, manager, config, allowed_tools=config["agent"]["allowed_tools"],
        user_message="继续",
    )
    assert action["tool"] == "allocate_mc_bohb"
    assert action["target_ids"] == ["B1"]


def test_relax_input_preparation_reuses_verified_migrated_results(tmp_path):
    manager, state, config = _completed_relax_fixture(tmp_path)
    old_version, new_version = "config-old", "config-new"
    config["mlip"].update({"name": "mace-mh-1", "version": "mace-mh-1",
                           "mace_head": "omat_pbe", "relax_parameters": {
                               "fmax": 0.05, "relax_steps": 150, "relax_cell": True,
                               "mace_default_dtype": "float64"}})
    state["confirmed_config_version"] = new_version
    state["config_migrations"] = [{
        "type": "approved_migration_with_verified_legacy_mace_defaults",
        "from": old_version, "to": new_version,
        "compatibility": {"basis": "verified_legacy_mace_mh_1_worker_defaults",
                          "model": "mace-mh-1", "legacy_config_version": old_version,
                          "verified_result_fields": {
                              "mace_head": "omat_pbe", "fmax_target_ev_per_angstrom": 0.05,
                              "cell_relaxed": True, "relax_steps_used_max": 150}}}]
    for task in state["tasks"]:
        task["config_version"] = old_version
        task["model_version"] = "mace-mh-1"
        task["outputs"].update({"mace_head": "omat_pbe",
                                "fmax_target_ev_per_angstrom": 0.05,
                                "cell_relaxed": True, "relax_steps_used": 150})
    config["supercomputer"] = {"worker": {"command": ["python3", "run_mlip_task.py"]}}
    result = prepare_relax_upload_batches(
        action={"target_ids": ["B1"], "parameters": {"selection_scope": "target_ids"}},
        context={"effective_config": config, "event_state": state, "manager": manager,
                 "phase_references": {}, "config_version": new_version},
    )
    assert result["status"] == "already_prepared"
    assert result["task_count"] == 0
    assert len(result["state"]["tasks"]) == 3
    assert not result["state"].get("pending_tasks")


def test_unverified_old_relax_result_does_not_skip_relax(tmp_path):
    manager, state, config = _completed_relax_fixture(tmp_path)
    config["mlip"].update({"name": "mace-mh-1", "version": "mace-mh-1",
                           "mace_head": "omat_pbe", "relax_parameters": {
                               "fmax": 0.05, "relax_steps": 150, "relax_cell": True,
                               "mace_default_dtype": "float64"}})
    for task in state["tasks"]:
        task["model_version"] = "mace-mh-1"
    action = choose_debug_next_action(state, manager, config,
        allowed_tools=config["agent"]["allowed_tools"], user_message="继续")

    assert action["tool"] == "prepare_local_batch_files"
    assert action["parameters"]["mode"] == "relax_inputs"


def test_budget_preview_is_read_only_and_exposes_full_plan(tmp_path):
    from decision_layer.strategy.estimate_branch_mc_budget import estimate_branch_mc_budget
    manager, state, config = _completed_relax_fixture(tmp_path)
    before = json.dumps(state, sort_keys=True)
    pool = state["branch_hull_batches"][state["current_branch_hull_version"]]
    preview = estimate_branch_mc_budget(list(manager.data["branches"].values()), pool, state,
        config, step_limit=1, seed=42)
    assert preview["full_plan_steps"] > preview["target_steps"]
    assert preview["compression_requires_user_choice"] is True
    assert preview["full_plan_branch_count"] == 1
    assert preview["full_plan_relative_cost"] > 0
    assert json.dumps(state, sort_keys=True) == before


def test_relax_phase_identification_preserves_branch(tmp_path):
    from analysis_layer.phase.identify_relax_result_phase import identify_relax_result_phase
    from pymatgen.core import Structure, Lattice
    path = tmp_path / "final.vasp"
    Structure(Lattice.cubic(4), ["Na", "O"], [[0, 0, 0], [.5, .5, .5]]).to(filename=str(path), fmt="poscar")
    manager = SimpleNamespace(data={"branches": {"B1": {"P": "O3"}}, "structures": {}})
    with patch("scientific_layer.structures.identify_branch.identify_phase", return_value={"phase": "P3"}):
        result = identify_relax_result_phase({"stage": "relax_and_feature", "status": "completed",
            "branch_id": "B1", "outputs": {"structure_path": str(path)}}, manager)
    assert result["branch_id"] == "B1"
    assert result["source_phase"] == "O3" and result["actual_phase"] == "P3"


def test_old_relax_approval_is_replaced_without_approving_mc(tmp_path):
    manager, state, config = _completed_relax_fixture(tmp_path)
    state["confirmed_config_version"] = "v1"
    state["confirmed_config"] = config
    old_action = {"tool": "prepare_local_batch_files", "task_key": "old-relax",
                  "target_ids": ["B1"], "parameters": {"mode": "relax_inputs"},
                  "budget": 1.0, "reason": "prepare Relax"}
    state["pending_execution_policies"] = {"review-1": {
        "record_id": "review-1",
        "agent_proposal": build_agent_proposal(old_action, {}), "revision": 0}}
    executed = []
    registry = create_tool_registry({
        "prepare_local_batch_files": lambda **kwargs: executed.append("relax"),
        "allocate_mc_bohb": _allocate_mc_bohb,
    })
    session = {"status": "confirmed", "confirmed_snapshot": {
        "config_version": "v1", "config": config}}
    context = {"manager": manager, "effective_config": config,
               "phase_references": {}, "event_state": state}
    result = run_tool_step(state, session, registry=registry,
        context=context,
        invocation_id="review-1", execution_mode="interactive",
        human_feedback={"decision": "approve", "comment": "同意"})
    assert result["status"] == "awaiting_approval"
    assert result["agent_proposal"]["raw_action"]["tool"] == "allocate_mc_bohb"
    assert result["revision"] == 1
    assert executed == []
    approved = run_tool_step(result["state"], session, registry=registry,
        context={**context, "event_state": result["state"]},
        invocation_id="review-1", execution_mode="interactive",
        human_feedback={"decision": "approve", "comment": "同意"})
    assert approved["status"] == "completed", approved.get("execution")
    assert approved["execution"]["result"]["mc_upload"]["status"] == "prepared"
    assert executed == []
    wait = summarize_manual_upload_wait(approved["state"], recovered_count=3)
    assert wait["waiting_by_stage"] == {"deep_search": 1}
    assert wait["upload_plan_path"] is None
    reply = format_workflow_reply({"status": "awaiting_manual_submission",
                                   "manual_wait": wait}, tmp_path / "state.json")
    assert "MC 任务" in reply
    assert "RELAX_UPLOAD_PLAN.json" not in reply

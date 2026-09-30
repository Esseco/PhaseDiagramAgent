from config_layer.session.apply_config_revision import apply_config_revision
from config_layer.session.confirm_config_snapshot import confirm_config_snapshot
from config_layer.session.create_config_draft import create_config_draft
from config_layer.defaults.default_layered_search_config import default_layered_search_config
from data_layer.ledger.phase_data_manager import PhaseDataManager
from run import run_workflow
import pytest


H = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def _session(boundary, *, selection_policy='finite_pool_kde'):
    draft = create_config_draft(default_layered_search_config(boundary=boundary))
    draft = apply_config_revision(draft, {
        "bohb.selection_policy": selection_policy,
        "calculation.mlip_version": "m1", "dft.parameters": {"encut": 520},
        "convergence.hull_tolerance": .01,
        "convergence.minimum_dft_validations": 1,
        "convergence.coverage_threshold": .9,
    })
    return confirm_config_snapshot(draft, user_confirmed=True)


def _bohb_action(task_key):
    return {
        "tool": "allocate_mc_bohb", "task_key": task_key, "budget": 0,
        "parameters": {"focus_regions": ["O3:x=1"], "exploration_fraction": .1,
                       "mc_budget": 90, "dft_budget": 0, "generation_quotas": {}},
        "reason": "search the recommended region",
    }


def test_public_entry_owns_bohb_selection_budget_and_resume(tmp_path):
    boundary = {"P": ["O3"], "H": {"O3": [H]}, "TM_ratio": {"Fe": 1}}
    manager = PhaseDataManager(boundary)
    branch_id = manager.add_branch(P="O3", H=H, x=1, T=["Fe"], composition={"Na": 1, "Fe": 1, "O": 2})
    session = _session(boundary)
    common = dict(
        manager=manager, phase_references={}, run_config={"state_path": str(tmp_path / "state.json")},
        config_session=session, execution_mode="autonomous", max_steps=1,
    )
    first = run_workflow(**common, agent_client=lambda _: _bohb_action("round-1"))
    assert first["status"] == "tasks_in_progress"
    task = first["state"]["pending_tasks"][0]
    assert task["branch_id"] == branch_id
    assert task["incremental_budget"] == 10
    assert task["planned_relative_cost"] == pytest.approx((4 / 40) ** 1.2)
    assert first["state"]["active_round"]["branch_selection_owner"] == "agent"
    assert first["state"]["active_round"]["mc_fidelity_owner"] == "hyperband"
    assert first["state"]["budget_reservations"][task["task_key"]]["stage"] == "deep_search"

    recovered = [{
        "task_id": task["task_id"], "task_key": task["task_key"], "status": "completed",
        "actual_cost": task["planned_relative_cost"], "minimum_energy_per_atom": -1.0,
        "group_reference_energy_per_atom": 0.0, "group_energy_scale": 1.0,
    }]
    second = run_workflow(
        **{**common, "state": first["state"]}, recovered_results=recovered,
        agent_client=lambda _: _bohb_action("round-2"),
    )
    promoted = second["state"]["pending_tasks"][0]
    assert promoted["budget"] == 30
    assert promoted["incremental_budget"] == 20
    assert promoted["selection_source"] == "bohb_promotion"
    assert second["state"]["budget_usage"]["total_relative_cost"] == pytest.approx(task["planned_relative_cost"])


def test_public_entry_uses_default_calculation_handler(tmp_path):
    boundary = {"P": ["O3"], "H": {"O3": [H]}, "TM_ratio": {"Fe": 1}}
    manager = PhaseDataManager(boundary); calls = []
    action = {"tool": "run_calculation_stage", "task_key": "check-1", "target_ids": ["S1"],
              "stage": "simple_check", "parameters": {}, "budget": .05, "reason": "screen"}
    result = run_workflow(
        manager, {}, {"state_path": str(tmp_path / "state.json")}, _session(boundary),
        dispatcher=lambda task: calls.append(task) or {**task, "status": "completed", "actual_cost": .03},
        agent_client=lambda _: action, execution_mode="autonomous", max_steps=1,
    )
    assert len(calls) == 1
    assert result["status"] == "completed"
    assert result["state"]["budget_usage"]["total_relative_cost"] == .03


def test_agent_cannot_take_over_bohb_branch_or_fidelity(tmp_path):
    boundary = {"P": ["O3"], "H": {"O3": [H]}, "TM_ratio": {"Fe": 1}}
    manager = PhaseDataManager(boundary)
    manager.add_branch(P="O3", H=H, x=1, T=["Fe"], composition={"Na": 1, "Fe": 1, "O": 2})
    action = _bohb_action("illegal"); action["parameters"]["branch_ids"] = ["B1"]
    result = run_workflow(
        manager, {}, {"state_path": str(tmp_path / "state.json")}, _session(boundary),
        agent_client=lambda _: action, execution_mode="autonomous", max_steps=1,
    )
    assert result["events"][0]["final_action"]["tool"] == "check_convergence"
    assert "bohb_owned_fields_forbidden" in result["events"][0]["final_action"]["fallback_reason"]
    assert "round_scheduler" not in result["state"]


def test_public_entry_hands_pending_tasks_to_runner(tmp_path):
    boundary = {"P": ["O3"], "H": {"O3": [H]}, "TM_ratio": {"Fe": 1}}
    manager = PhaseDataManager(boundary)
    manager.add_branch(P="O3", H=H, x=1, T=["Fe"], composition={"Na": 1, "Fe": 1, "O": 2})

    class Runner:
        def __init__(self): self.prepared = []
        def collect_results(self, state): return []
        def prepare(self, state):
            self.prepared = [item["task_id"] for item in state["pending_tasks"]]
            return {"status": "prepared", "state": state, "batch": {"task_ids": self.prepared}}

    runner = Runner()
    result = run_workflow(
        manager, {}, {"state_path": str(tmp_path / "state.json")}, _session(boundary),
        agent_client=lambda _: _bohb_action("round-slurm"), execution_mode="autonomous",
        max_steps=1, task_runner=runner,
    )
    assert result["status"] == "tasks_prepared"
    assert runner.prepared == [result["state"]["pending_tasks"][0]["task_id"]]
    assert result["batch"]["task_ids"] == runner.prepared


def test_public_entry_creates_budgeted_dft_child_tasks_from_qbc(tmp_path):
    boundary = {"P": ["O3"], "H": {"O3": [H]}, "TM_ratio": {"Fe": 1}}
    manager = PhaseDataManager(boundary)
    branch_id = manager.add_branch(P="O3", H=H, x=1, T=["Fe"], composition={"Na": 1, "Fe": 1, "O": 2})
    structure_id = manager.add_structure(
        branch_id=branch_id, arrangement={"order": 1}, metadata={"atom_count": 40}
    )
    state = {"qbc_candidates": [{
        "candidate_id": structure_id, "branch_id": branch_id, "atom_count": 40,
        "qbc": {"status": "completed", "f_std_max": .3, "energy_std": .02},
    }]}
    action = {
        "tool": "select_dft_candidates", "task_key": "dft-select-1",
        "target_ids": [structure_id], "budget": 0,
        "parameters": {"decisions": [{
            "candidate_id": structure_id, "action": "DFT_SINGLE_POINT", "reason": "high disagreement"
        }], "global_action": "CONTINUE_DATA_COLLECTION"},
        "reason": "label uncertain near-hull candidate",
    }
    result = run_workflow(
        manager, {}, {"state_path": str(tmp_path / "state.json")}, _session(boundary),
        state=state, agent_client=lambda _: action, execution_mode="autonomous", max_steps=1,
    )
    assert result["status"] == "tasks_in_progress"
    task = result["state"]["pending_tasks"][0]
    assert task["stage"] == "dft_single_point"
    assert task["structure_id"] == structure_id
    assert task["parameters"] == {"encut": 520}
    assert result["state"]["budget_reservations"][task["task_key"]]["reserved_cost"] == 30.0


def test_recovered_dft_without_final_structure_updates_ledger_but_not_hull(tmp_path):
    boundary = {"P": ["O3"], "H": {"O3": [H]}, "TM_ratio": {"Fe": 1}}
    manager = PhaseDataManager(boundary)
    branch = manager.add_branch(P="O3", H=H, x=1, T=["Fe"], composition={"Fe": 1})
    structure = manager.add_structure(branch_id=branch, arrangement={"order": 1}, composition={"Fe": 1})
    task = {
        "task_id": "DFT1", "task_key": "DK1", "structure_id": structure,
        "object_id": structure, "stage": "dft_single_point", "status": "pending",
    }
    state = {
        "tasks": [task], "pending_tasks": [task],
        "budget_reservations": {"DK1": {
            "task_key": "DK1", "stage": "dft_single_point", "status": "reserved",
            "reserved_cost": 30.0,
        }},
    }
    recovered = [{
        **task, "status": "completed", "converged": True, "actual_cost": 28.0,
        "outputs": {"energy": -8.0, "final_frame_mlip_energy": -7.998, "energy_unit": "eV", "dft_code": "VASP", "dft_version": "6.5.1"},
        "result_path": str(tmp_path / "dft"),
    }]
    result = run_workflow(
        manager, {}, {
            "state_path": str(tmp_path / "state.json"),
            "ledger_path": str(tmp_path / "ledger.json"),
            "phase_diagram_directory": str(tmp_path / "hulls"),
        }, _session(boundary), state=state, recovered_results=recovered,
        agent_client=None, execution_mode="autonomous", max_steps=1,
    )
    history = manager.data["structures"][structure]["stage_history"]["dft_single_point"]
    assert len(history) == 1
    assert result["state"]["phase_diagrams"]["dft"]["status"] == "unknown"
    assert result["state"]["phase_diagrams"]["dft"]["reason"] == "no_identified_energy_records"
    assert not result["state"]["phase_diagrams"]["dft"]["entries"]
    record = result["state"]["phase_records"][0]
    assert record["phase_identification_status"] == "unknown"
    assert record["phase_identification"]["reason"] == "final_structure_missing"
    assert result["state"]["coverage"]
    assert result["state"]["agent_state_summary"]["completed_task_keys"] == ["DK1"]
    assert result["state"]["budget_usage"]["total_relative_cost"] == 28.0
    assert result["state"]["final_frame_dft_errors"][0]["error_ev_per_atom"] == pytest.approx(.002)
    assert (tmp_path / "ledger.json").is_file()

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock
import sys

import pytest


def bad_task():
    return {
        "task_id": "bad",
        "task_key": "bad-key",
        "stage": "relax_and_feature",
        "status": "completed",
        "converged": True,
        "structure_id": "s2",
        "model_version": "new",
        "model_refresh_id": "p",
        "input_structure_sha256": "s2",
        "parameters": {"model_refresh_operation": "relax", "relax_steps": 150},
        "outputs": {
            "optimizer_converged": False,
            "relax_stopped_normally": True,
            "force_max_ev_per_angstrom": 0.2,
            "fmax_target_ev_per_angstrom": 0.05,
            "energy": -100,
            "energy_unit": "eV",
            "actual_phase": "O3",
            "phase_identification": {"status": "identified", "phase": "O3"},
        },
    }


@pytest.mark.parametrize(
    "operation,physical,expected",
    [("relax", False, False), ("relax", True, True), ("mc", False, True), ("predict", None, None)],
)
def test_executor_uses_physical_convergence_only_for_relax(
    monkeypatch, operation, physical, expected
):
    from phase_agent.science.mlip.slurm_executor import execute_mlip_task

    returned = {
        "status": "completed",
        "relax_stopped_normally": operation != "predict",
        "optimizer_converged": physical,
        "force_max_ev_per_angstrom": 0.01 if physical else 0.2,
        "fmax_target_ev_per_angstrom": 0.05,
    }
    if operation == "predict":
        returned["single_point_completed"] = True
    worker = Mock(return_value=returned)
    monkeypatch.setattr("phase_agent.science.mlip.slurm_executor.run_mace_worker", worker)
    task = {
        "stage": "deep_search" if operation == "mc" else "relax_and_feature",
        "structure_id": "s",
        "result_path": "unused/result.json",
        "worker_job": {
            "operation": operation,
            "structure_path": "initial.vasp",
            "model_path": "fake.model",
        },
    }
    result = execute_mlip_task(task)
    assert result["converged"] is expected
    if operation == "relax" and not physical:
        assert result["checks_passed"] is False
    worker.assert_called_once()


def test_legacy_repair_preserves_evidence_costs_and_cannot_be_reactivated():
    from phase_agent.analysis.convergence.mlip_relaxation import reconcile_relaxation_quality
    from phase_agent.analysis.phase.refresh_identified_phases import refresh_identified_phases

    state = {
        "tasks": [bad_task()],
        "phase_records": [
            {
                "record_id": "bad-record",
                "source_task_id": "bad",
                "structure_id": "s2",
                "source_version": "new",
                "energy_method": "mlip",
                "energy": -100,
                "phase": "O3",
                "status": "completed",
            }
        ],
        "cost_history": [{"gpu_hours": 2}],
        "budget_usage": {"relax_and_feature": {"used_cost": 10}},
    }
    old = deepcopy(state)
    current = reconcile_relaxation_quality(state)
    assert state == old
    assert current["tasks"][0]["converged"] is False
    assert current["tasks"][0]["status"] == "completed"
    assert current["phase_records"][0]["status"] == "excluded_unconverged_relaxation"
    assert current["tasks"][0]["outputs"] == old["tasks"][0]["outputs"]
    assert (
        current["cost_history"] == old["cost_history"]
        and current["budget_usage"] == old["budget_usage"]
    )
    refresh_identified_phases(current)
    assert current["phase_records"][0]["status"] == "excluded_unconverged_relaxation"
    assert reconcile_relaxation_quality(current) == current


def test_cell_filter_convergence_is_not_rejected_by_raw_force_transformation():
    from phase_agent.analysis.convergence.mlip_relaxation import relaxation_failure

    task = bad_task()
    task["outputs"]["optimizer_converged"] = True
    assert relaxation_failure(task) is None
    task["outputs"].pop("optimizer_converged")
    assert relaxation_failure(task) == "relaxation_force_above_target"


def test_repair_invalidates_only_bad_pools_and_keeps_a_rebuilt_clean_pool():
    from phase_agent.analysis.convergence.mlip_relaxation import reconcile_relaxation_quality

    task = bad_task()
    task["outputs"]["structure_path"] = "bad.vasp"
    state = {
        "tasks": [task],
        "branch_hull_batches": {
            "old": {"records": [{"structure_path": "bad.vasp"}]},
            "clean": {"records": [{"structure_path": "good.vasp"}]},
        },
        "current_branch_hull_by_model": {"new": "old"},
        "current_branch_hull_version": "old",
    }
    current = reconcile_relaxation_quality(state)
    assert current["current_branch_hull_version"] is None
    current.update(
        current_branch_hull_version="clean", current_branch_hull_by_model={"new": "clean"}
    )
    assert reconcile_relaxation_quality(current) == current


def test_relax_pool_does_not_reuse_flattened_false_convergence():
    from phase_agent.analysis.phase.branch_relax_hull import build_relax_hull

    row = {
        "model_version": "new",
        "stage": "relax_and_feature",
        "converged": True,
        "optimizer_converged": False,
        "energy_unit": "eV",
        "energy": -10,
    }
    assert build_relax_hull([row], model_version="new")["records"] == []


def test_frozen_hull_with_discarded_results_cannot_be_reused(monkeypatch):
    import hashlib
    import json
    from phase_agent.tools.workflows.prepare_branch_relaxation import prepare_branch_relaxation

    bad = bad_task()
    bad["outputs"]["structure_path"] = "bad.vasp"
    settings = hashlib.sha256(json.dumps({}, sort_keys=True).encode()).hexdigest()[:12]
    good = {
        "task_id": "good",
        "task_key": f"relax-screen:new:{settings}:s",
        "stage": "relax_and_feature",
        "status": "completed",
        "converged": True,
        "model_version": "new",
        "structure_id": "s",
        "outputs": {
            "composition": {"Co": 1, "O": 2},
            "structure_path": "good.vasp",
            "energy": -3,
            "energy_unit": "eV",
            "optimizer_converged": True,
        },
    }
    state = {
        "tasks": [bad, good],
        "branch_hull_batches": {
            "frozen": {"model_version": "new", "records": [{"structure_path": "bad.vasp"}]}
        },
    }
    manager = SimpleNamespace(
        data={
            "structures": {
                "s": {
                    "metadata": {
                        "initialization_method": "electrostatic_top10_random3_layer_occupied"
                    }
                }
            }
        }
    )
    monkeypatch.setattr(
        "phase_agent.tools.workflows.prepare_branch_relaxation._screen_structure_ids",
        lambda *a, **k: ["s"],
    )
    result = prepare_branch_relaxation(
        [{"branch_id": "b", "structure_ids": ["s"]}],
        state,
        {
            "manager": manager,
            "effective_config": {
                "bohb": {"relax_structures_per_branch": 1},
                "mlip": {"version": "new"},
            },
            "mc_hull_reference_version": "frozen",
        },
    )
    assert result["reason"] == "approved_hull_contains_unconverged_results"
    assert len(result["state"]["tasks"]) == 2


def test_rejected_relaxation_cannot_create_a_phase_record():
    from phase_agent.persistence.ledger.collect_calculation_results import _phase_record

    assert (
        _phase_record(SimpleNamespace(data={}), "s2", "r", "relax_and_feature", bad_task()) is None
    )


def test_discarded_refresh_results_finish_as_partial_not_waiting_forever():
    from tests.test_refresh_completion_reply import refreshed_state
    from phase_agent.analysis.convergence.mlip_relaxation import reconcile_relaxation_quality
    from phase_agent.tools.workflows.model_refresh_gate import run_model_refresh_gate

    state = refreshed_state()
    state["tasks"].append(bad_task())
    state["model_refresh"]["task_ids"].append("bad")
    state["model_refresh"]["plan"]["total_unique"] = 2
    state = reconcile_relaxation_quality(state)
    result = run_model_refresh_gate(state, {}, registry={}, context={})
    assert result["status"] == "completed"
    assert result["completed_refresh"]["discarded_unconverged_count"] == 1
    assert result["completed_refresh"]["deferred"] == 1
    assert result["state"]["model_refresh"]["refreshed"] == 1


def test_hull_rejects_mlip_quality_failure_even_when_status_says_completed():
    from phase_agent.analysis.phase.update_phase_diagram import update_phase_diagram

    records = [
        {
            "record_id": "zero",
            "composition": {"Co": 1, "O": 2},
            "energy": -3,
            "energy_unit": "eV",
            "energy_method": "mlip",
            "source_version": "new",
            "phase": "O1",
        },
        {
            "record_id": "one",
            "composition": {"Na": 1, "Co": 1, "O": 2},
            "energy": -4,
            "energy_unit": "eV",
            "energy_method": "mlip",
            "source_version": "new",
            "phase": "O3",
        },
        {
            "record_id": "bad",
            "composition": {"Na": 0.5, "Co": 1, "O": 2},
            "energy": -100,
            "energy_unit": "eV",
            "energy_method": "mlip",
            "source_version": "new",
            "phase": "O3",
            "checks_passed": False,
        },
    ]
    diagram = update_phase_diagram(records, active_model_version="new", include_dft=False)[
        "diagrams"
    ]["mlip"]
    assert "bad" not in {e["record_id"] for e in diagram["entries"]}
    assert any(r["record_id"] == "bad" for r in diagram["rejected"])


def test_relax_worker_defaults_to_300_lbfgs_without_loading_a_model(tmp_path, monkeypatch):
    from pymatgen.core import Lattice, Structure
    from phase_agent.science.mlip.mace_worker import run_mace_worker

    seen = {}

    def relax(structure, model, *, output_path, steps, **kwargs):
        seen["steps"] = steps
        Structure(
            Lattice.cubic(5), ["Co", "O", "O"], [[0, 0, 0], [0.2, 0.2, 0.2], [0.4, 0.4, 0.4]]
        ).to(filename=output_path, fmt="poscar")
        return {"optimizer_converged": False, "relax_stopped_normally": True}

    monkeypatch.setitem(
        sys.modules, "Process_AL_MC.relax", SimpleNamespace(relax_structure_mace=relax)
    )
    result = run_mace_worker(
        {
            "operation": "relax",
            "model_path": "fake.model",
            "structure_path": "unused.vasp",
            "output_directory": str(tmp_path),
            "parameters": {"optimizer": "LBFGS"},
        }
    )
    assert seen["steps"] == 300 and result["optimizer"] == "LBFGS"


def test_unconverged_refresh_does_not_spend_more_committee_predictions(tmp_path, monkeypatch):
    from pymatgen.core import Lattice, Structure
    from phase_agent.science.mlip.mace_worker import run_mace_worker

    def relax(structure, model, *, output_path, **kwargs):
        Structure(
            Lattice.cubic(5), ["Co", "O", "O"], [[0, 0, 0], [0.2, 0.2, 0.2], [0.4, 0.4, 0.4]]
        ).to(filename=output_path, fmt="poscar")
        return {"optimizer_converged": False, "relax_stopped_normally": True}

    monkeypatch.setitem(
        sys.modules, "Process_AL_MC.relax", SimpleNamespace(relax_structure_mace=relax)
    )
    forbidden = Mock(side_effect=AssertionError("No further predictions after failed Relax"))
    monkeypatch.setattr("phase_agent.science.mlip.predict_structure.predict_structure", forbidden)
    result = run_mace_worker(
        {
            "operation": "relax",
            "model_path": "fake",
            "structure_path": "unused",
            "output_directory": str(tmp_path),
            "parameters": {"model_refresh_operation": "relax"},
        }
    )
    assert result["optimizer_converged"] is False
    forbidden.assert_not_called()

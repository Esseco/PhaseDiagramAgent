from analysis_layer.feedback.export_dft_products import dft_product_path
"""Scoped spin proxy changes scientific eligibility, never execution facts."""
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pymatgen.core import Lattice, Structure
from scientific_layer.dft.spin_acceptance import apply_dft_spin_standard
from scientific_layer.training.prepare_mace_finetune import prepare_mace_finetune
from analysis_layer.feedback.dft_result_products import record_dft_products, export_dft_products
from data_layer.ledger.collect_calculation_results import collect_calculation_results


def make_result(stage="dft_single_point", moments=(4.3, -3.9, 0, 0)):
    struct = Structure(Lattice.cubic(4), ["Fe", "Mn", "O", "O"],
                       [[0,0,0],[.5,.5,0],[0,.5,.5],[.5,0,.5]])
    outputs = {"structure": struct.as_dict(), "composition": struct.composition.get_el_amt_dict(),
               "actual_phase": "O3", "phase_identification": {"status": "identified", "phase": "O3"},
               "training_ready": True, "energy": -10, "energy_unit": "eV", "training_energy": -10,
               "atom_count": 4, "forces": [[0,0,0]]*4, "forces_unit": "eV/angstrom",
               "magnetic_check": {"elements": ["Fe", "Mn", "O", "O"],
                   "moments": list(moments) if moments is not None else None, "is_layered_oxide": True}}
    return {"task_id": "D1", "task_key": "K1", "structure_id": "S1", "stage": stage,
            "status": "completed", "converged": True, "checks_passed": True,
            "actual_cost": 30, "model_version": "m1", "outputs": outputs}


def make_manager():
    stages = ["dft_single_point", "dft_relax"]
    return SimpleNamespace(STAGES=stages, stages=stages, data={
        "system_config": {"system_id": "layered_na_tm_oxide"},
        "structures": {"S1": {"branch_id": "B1", "composition": {"Fe": 1,"Mn": 1,"O": 2}}},
        "branches": {"B1": {"P": "O3"}}}, record_result=Mock(return_value="R1"))


@pytest.mark.parametrize("stage", ["dft_single_point", "dft_relax"])
@pytest.mark.parametrize("moments,expected", [((4.3,-3.9,0,0),"passed"), ((1.0,3.9,0,0),"rejected"), (None,"unknown")])
def test_gate_scope_and_execution_preserved(stage, moments, expected):
    incoming = make_result(stage, moments)
    before = deepcopy(incoming)
    checked = apply_dft_spin_standard(incoming, make_manager())
    assert incoming == before
    assert checked["outputs"]["spin_state_check"]["status"] == expected
    assert checked["outputs"]["spin_state_check"]["magnetic_ground_state_proven"] is False
    assert checked["checks_passed"] is (expected == "passed")
    assert checked["actual_cost"] == 30 and checked["status"] == "completed"
    assert checked["outputs"]["energy"] == -10


def test_nonlayered_and_other_elements_unchanged():
    manager = make_manager()
    manager.data["system_config"] = {"system_id": "other_system"}
    incoming = make_result(moments=None)
    incoming["outputs"].update(actual_phase="spinel", phase_identification={})
    incoming["outputs"]["magnetic_check"]["is_layered_oxide"] = False
    assert apply_dft_spin_standard(incoming, manager)["checks_passed"] is True
    incoming = make_result(moments=None)
    struct = Structure(Lattice.cubic(4), ["Ni", "O"], [[0,0,0],[.5,.5,.5]])
    incoming["outputs"].update(structure=struct.as_dict(), composition={"Ni": 1,"O": 1}, magnetic_check={})
    checked = apply_dft_spin_standard(incoming, make_manager())
    assert checked["outputs"]["spin_state_check"]["status"] == "not_applicable"
    assert checked["checks_passed"] is True


def test_spin_pass_cannot_overwrite_other_quality_failure():
    incoming = make_result()
    incoming["checks_passed"] = False
    checked = apply_dft_spin_standard(incoming, make_manager())
    assert checked["outputs"]["spin_state_check"]["status"] == "passed"
    assert checked["checks_passed"] is False


@pytest.mark.parametrize("moments", [None, (1,3.9,0,0)])
def test_bad_spin_preserved_but_not_hull_training_or_metrics(tmp_path, moments):
    manager = make_manager()
    incoming = make_result(moments=moments)
    collected = collect_calculation_results(manager, "S1", apply_dft_spin_standard(incoming, manager))
    assert collected["status"] == "completed"
    assert collected["result_id"] is None and collected["phase_record"] is None
    manager.record_result.assert_not_called()
    assert manager.data["structures"]["S1"]["metadata"]["calculation_attempts"][0]["checks_passed"] is False
    state = {"tasks": [{"task_id": "D1", "stage": incoming["stage"], "status": "completed", "model_version": "m1"}]}
    evaluator = Mock(side_effect=AssertionError("not eligible for evaluation"))
    record_dft_products(state, incoming, evaluator, manager)
    evaluator.assert_not_called()
    assert len(state["dft_training_records"]) == 1  # raw labels retained for audit
    assert state["dft_training_records"][0]["checks_passed"] is False
    assert not state.get("new_dft_records")
    prepared = prepare_mace_finetune(state["dft_training_records"], tmp_path / "fine", config={})
    assert prepared["accepted"] == 0
    export_dft_products(state, tmp_path / "diagrams")
    from pathlib import Path
    root = Path(next(iter(state["dft_result_exports"].values()))["directory"])
    assert json.loads((dft_product_path(root, "training.json")).read_text()) == []
    records = json.loads((dft_product_path(root, "dft_records.json")).read_text())
    assert records[0]["energy"] == -10
    metrics = json.loads((dft_product_path(root, "mlip_dft_metrics.json")).read_text(encoding="utf-8"))
    assert metrics["recovered_tasks"] == 1 and metrics["matched_structures"] == 0
    assert metrics["energy_total"]["mae"] is None


def test_good_spin_enters_ledger_and_training():
    manager = make_manager()
    collected = collect_calculation_results(manager, "S1", make_result())
    assert collected["result_id"] == "R1"
    assert collected["phase_record"]["energy"] == -10
    state = {}
    record_dft_products(state, make_result(), manager=manager)
    assert state["dft_training_records"][0]["checks_passed"] is True

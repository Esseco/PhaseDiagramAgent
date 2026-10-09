from analysis_layer.feedback.export_dft_products import dft_product_path
import json
from copy import deepcopy

import numpy as np
import pytest
from pymatgen.core import Structure, Lattice

from analysis_layer.feedback.dft_result_products import record_dft_products, export_dft_products
from execution_layer.workflows.create_dft_comparison_evaluator import create_dft_comparison_evaluator
from scientific_layer.dft.parse_vasp_result import parse_vasp_result
from types import SimpleNamespace


def result():
    structure = Structure(Lattice.cubic(4), ["Na", "O"], [[0, 0, 0], [.5, .5, .5]])
    return {"task_id": "T1", "structure_id": "S1", "stage": "dft_single_point",
            "status": "completed", "converged": True, "model_version": "m1",
            "outputs": {"structure": structure.as_dict(), "composition": {"Na": 1, "O": 1},
                        "energy": -8., "training_energy": -8., "energy_unit": "eV",
                        "forces": [[0., 0., 0.], [0., 0., 0.]], "forces_unit": "eV/angstrom",
                        "atom_count": 2, "training_ready": True}}


def predict(**kwargs):
    return {"model_version": "m1", "energy": -7., "forces": [[.2]*3]*2,
            "energy_unit": "eV", "forces_unit": "eV/angstrom"}


def test_round_products_and_units(tmp_path):
    state = {"tasks": [{"task_id": "T1", "upload_operation_id": "dft-1"}]}
    record_dft_products(state, result(), predict)
    assert state["new_dft_records"][0]["structure"]["sites"]
    export_dft_products(state, tmp_path)
    directory = next(iter(state["dft_result_exports"].values()))["directory"]
    from pathlib import Path
    metrics = json.loads((dft_product_path(directory, "mlip_dft_metrics.json")).read_text(encoding="utf-8"))
    assert metrics["energy_total"]["mae"] == 1
    assert metrics["energy_per_atom"]["rmse"] == .5
    assert metrics["forces"]["mae"] == pytest.approx(.2)
    assert metrics["forces"]["rmse"] == pytest.approx(.2)
    for field in ("energy_total", "energy_per_atom", "forces"):
        assert "mse" not in metrics[field] and "mse_unit" not in metrics[field]
        assert metrics[field]["rmse_unit"] == metrics[field]["mae_unit"]
    assert metrics["round_scope"]["upload_operation_id"] == "dft-1"
    timestamp = (dft_product_path(directory, "training.json")).stat().st_mtime_ns
    export_dft_products(state, tmp_path)
    assert (dft_product_path(directory, "training.json")).stat().st_mtime_ns == timestamp


def test_model_mismatch_not_counted_but_training_preserved():
    state = {}
    record_dft_products(state, result(), lambda **kw: {**predict(), "model_version": "m2"})
    assert state["dft_mlip_comparisons"][0]["status"] == "not_evaluated"
    assert len(state["new_dft_records"]) == 1


def test_rmse_aggregates_squared_errors_not_frame_rmse(tmp_path):
    from pathlib import Path
    state = {"tasks": [{"task_id": name, "upload_operation_id": "round1"} for name in ("T1", "T2")]}
    record_dft_products(state, result(), predict)
    second = deepcopy(result())
    second["task_id"] = "T2"
    def other_prediction(**kwargs):
        return {**predict(), "energy": -5., "forces": [[.4]*3]*2}
    record_dft_products(state, second, other_prediction)
    export_dft_products(state, tmp_path)
    directory = Path(next(iter(state["dft_result_exports"].values()))["directory"])
    metrics = json.loads(dft_product_path(directory, "mlip_dft_metrics.json").read_text(encoding="utf-8"))
    assert metrics["energy_total"]["mae"] == 2
    assert metrics["energy_total"]["rmse"] == pytest.approx(np.sqrt(5))
    assert metrics["forces"]["rmse"] == pytest.approx(np.sqrt(.1))


def test_default_evaluator_refuses_wrong_epoch():
    evaluator = create_dft_comparison_evaluator({"mlip": {"version": "m2"}})
    with pytest.raises(ValueError, match="version"):
        evaluator(result=result(), structure_id="S1", manager=None)


def test_vasp_parser_populates_portable_frame(tmp_path):
    (tmp_path / "vasprun.xml").write_text("stub")
    structure = Structure.from_dict(result()["outputs"]["structure"])
    parsed = SimpleNamespace(converged=True, final_energy=-9., ionic_steps=[{
        "structure": structure, "e_0_energy": -8., "forces": [[0, 0, 0]]*2}])
    output = parse_vasp_result(tmp_path, parser=lambda path: parsed)["outputs"]
    assert output["energy"] == -8. and output["training_ready"] is True
    assert output["composition"] == {"Na": 1., "O": 1.}


def test_official_feedback_is_idempotent(tmp_path):
    from unittest.mock import patch
    from execution_layer.workflows.apply_scientific_feedback import apply_scientific_feedback
    manager = SimpleNamespace(STAGES=["dft_single_point"], stages=["dft_single_point"],
                              stage_labels={}, data={"structures": {"S1": {}}, "branches": {}})
    module = "execution_layer.workflows.apply_scientific_feedback."
    with patch(module + "collect_calculation_results", return_value={"phase_record": None}), \
         patch(module + "ensure_phase_identification", side_effect=lambda state, *a, **kw: (state, {})), \
         patch(module + "coverage", return_value={}):
        first = apply_scientific_feedback({}, [result()], manager=manager,
                    phase_diagram_directory=tmp_path, final_frame_mlip_evaluator=predict)
        second = apply_scientific_feedback(first["state"], [result()], manager=manager,
                    phase_diagram_directory=tmp_path, final_frame_mlip_evaluator=predict)
    assert len(second["state"]["new_dft_records"]) == 1
    assert len(second["state"]["dft_mlip_comparisons"]) == 1
    assert second["state"]["dft_result_exports"]
    directory = next(iter(second["state"]["dft_result_exports"].values()))["directory"]
    from pathlib import Path
    assert all((dft_product_path(directory, name)).is_file() for name in (
        "energy_comparison.csv", "force_comparison.csv", "metrics.csv"))

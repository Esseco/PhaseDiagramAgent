from types import SimpleNamespace

import numpy as np
from pymatgen.core import Structure, Lattice

from phase_agent.science.dft.vasp_training_labels import extract_training_frames
from phase_agent.science.dft.parse_vasp_result import parse_vasp_result
from phase_agent.analysis.feedback.dft_result_products import record_dft_products, export_dft_products
from phase_agent.science.training.prepare_mace_finetune import prepare_mace_finetune
from phase_agent.science.training.update_mlip import _unique


def parsed_relax():
    structure = Structure(Lattice.cubic(4), ["Na", "O"], [[0, 0, 0], [.5, .5, .5]])
    frames = [{"structure": structure, "e_0_energy": float(-8-i),
               "forces": [[.1*i, 0, 0], [-.1*i, 0, 0]],
               "electronic_steps": [{}]*count} for i, count in enumerate((2, 5, 3))]
    frames[0]["stress"] = np.eye(3)*10
    return SimpleNamespace(converged=False, converged_ionic=False, parameters={"NELM": 5},
                           ionic_steps=frames, final_energy=-10., vasp_version="6")


def test_chgnet_style_frames_do_not_require_ionic_convergence():
    extracted = extract_training_frames(parsed_relax())
    assert [f["training_frame_index"] for f in extracted["frames"]] == [0, 2]
    assert extracted["rejected"][0]["frame_index"] == 1
    assert "stress" not in extracted["frames"][1]
    np.testing.assert_allclose(extracted["frames"][0]["stress"], -np.eye(3)*.006241509074460763)


def test_relax_training_only_no_final_hull_metrics(tmp_path):
    (tmp_path / "vasprun.xml").write_text("stub")
    result = parse_vasp_result(tmp_path, parser=lambda path: parsed_relax())
    result.update(task_id="R1", structure_id="S1", stage="dft_relax", model_version="m1")
    assert result["status"] == "failed" and result["converged"] is False
    assert len(result["outputs"]["training_frames"]) == 2
    assert not (tmp_path / "final_structure.vasp").exists()
    assert result["outputs"]["final_frame_index"] == 2
    state = {}
    def must_not_evaluate(**kwargs):
        raise AssertionError("unconverged relaxation compared as final structure")
    record_dft_products(state, result, must_not_evaluate)
    assert len(state["new_dft_records"]) == 2
    assert len(_unique(state["new_dft_records"])) == 2
    assert state["dft_mlip_comparisons"][0]["status"] == "not_evaluated"
    assert all(r["electronic_converged"] and not r["ionic_converged"] for r in state["new_dft_records"])
    report = prepare_mace_finetune(state["new_dft_records"], tmp_path / "training",
                                  config={"split": {"train": 1, "valid": 0, "test": 0}})
    assert report["accepted"] == 2
    export_dft_products(state, tmp_path / "products")


def test_bad_final_frame_never_replaced_by_previous_for_final_labels(tmp_path):
    parsed = parsed_relax()
    parsed.ionic_steps[-1]["electronic_steps"] = [{}]*5
    (tmp_path / "vasprun.xml").write_text("stub")
    result = parse_vasp_result(tmp_path, parser=lambda path: parsed)
    assert result["outputs"]["training_ready"] is False
    assert [f["training_frame_index"] for f in result["outputs"]["training_frames"]] == [0]


def test_partial_xml_never_reports_completed(tmp_path):
    from unittest.mock import patch
    from lxml.etree import XMLSyntaxError
    parsed = parsed_relax()
    parsed.converged = True
    (tmp_path / "vasprun.xml").write_text("partial")
    with patch("pymatgen.io.vasp.outputs.Vasprun", side_effect=[
        XMLSyntaxError("truncated", 0, 0, 0), parsed]):
        result = parse_vasp_result(tmp_path)
    assert result["status"] == "failed"
    assert result["outputs"]["xml_complete"] is False
    assert len(result["outputs"]["training_frames"]) == 2


def test_nonzero_exit_preserves_verified_frames(tmp_path):
    (tmp_path / "vasprun.xml").write_text("stub")
    result = parse_vasp_result(tmp_path, exit_code=9, parser=lambda p: parsed_relax())
    assert result["status"] == "failed" and "code 9" in result["error"]
    assert len(result["outputs"]["training_frames"]) == 2


def test_checkpoint_can_resolve_one_nested_atomate_calculation(tmp_path):
    import json
    from phase_agent.tools.remote.finalize_vasp import resolve_calculation_directory
    (tmp_path / "workflow.json").write_text(json.dumps({"calculation": "relax"}))
    (tmp_path / "workflow_state.json").write_text(json.dumps({"stages": {
        "relax": {"directory": "runs/current"}}}))
    calculation = tmp_path / "runs/current/job_1"
    calculation.mkdir(parents=True)
    (calculation / "vasprun.xml.gz").write_text("stub")
    assert resolve_calculation_directory(tmp_path) == calculation

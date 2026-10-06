"""Portable final-frame magnetic evidence remains diagnostic, not an execution gate."""
from types import SimpleNamespace
from unittest.mock import patch
from pymatgen.core import Lattice, Structure

from scientific_layer.dft.magnetic_diagnostics import diagnose_result_magnetism, extract_magnetic_diagnostics
from analysis_layer.feedback.dft_result_products import record_dft_products


def result(elements=("Fe", "Mn", "O"), moments=(4.3, 2.1, 0)):
    struct = Structure(Lattice.cubic(4), list(elements), [[0,0,0],[.5,.5,0],[0,.5,.5]][:len(elements)])
    return {"task_id": "D1", "structure_id": "S1", "stage": "dft_relax", "status": "completed",
        "converged": True, "model_version": "m1", "outputs": {"structure": struct.as_dict(),
        "energy": -9, "training_ready": True, "magnetic_check": {"moments": list(moments),
        "elements": list(elements), "is_layered_oxide": None}}}


def manager():
    return SimpleNamespace(data={"system_config": {"system_id": "layered_na_tm_oxide", "is_layered_oxide": True}, "structures": {}})


def test_local_warning_retained_and_cached():
    incoming = result()
    checked = diagnose_result_magnetism(incoming, manager())
    report = checked["outputs"]["magnetic_check"]
    assert report["status"] == "warning"
    assert report["anomalous_atoms"][0]["element"] == "Mn"
    assert incoming["outputs"]["magnetic_check"].get("status") is None
    assert checked["status"] == "completed" and checked["outputs"]["energy"] == -9
    with patch("Process_Vasp.check_layered_oxide_moments", side_effect=AssertionError("must reuse")):
        assert diagnose_result_magnetism(checked, manager()) == checked
    state = {}
    record_dft_products(state, checked)
    assert state["dft_dataset_records"][0]["magnetic_check"] == report
    assert state["dft_training_records"][0]["magnetic_check"] == report


def test_only_supported_elements_and_no_na_required():
    assert diagnose_result_magnetism(result(("Ni", "O"), (0,0)), manager())["outputs"]["magnetic_check"]["status"] == "skipped"
    assert diagnose_result_magnetism(result(("Fe", "O"), (-4.3,0)), manager())["outputs"]["magnetic_check"]["status"] == "passed"


def test_missing_is_unknown_and_phase_can_establish_layered():
    incoming = result()
    incoming["outputs"]["magnetic_check"] = {}
    assert diagnose_result_magnetism(incoming, manager())["outputs"]["magnetic_check"]["status"] == "unknown"
    incoming["outputs"]["magnetic_check"]["moments"] = [4.3,3.9,0]
    incoming["outputs"]["actual_phase"] = "O3"
    assert diagnose_result_magnetism(incoming)["outputs"]["magnetic_check"]["status"] == "passed"


def test_remote_missing_reader_doesnt_change_energy_labels(tmp_path):
    struct = Structure.from_dict(result()["outputs"]["structure"])
    with patch("Process_Vasp.read_dft_magnetic_data", side_effect=OSError("missing OUTCAR")):
        report = extract_magnetic_diagnostics(tmp_path, SimpleNamespace(final_structure=struct, ionic_steps=[{}]))
    assert report["status"] == "unknown"


def test_site_and_frame_mismatch_not_passed():
    incoming = result()
    incoming["outputs"]["magnetic_check"]["elements"] = ["Mn", "Fe", "O"]
    assert diagnose_result_magnetism(incoming, manager())["outputs"]["magnetic_check"]["status"] == "unknown"
    incoming = result()
    incoming["outputs"].update(final_frame_index=1, training_frame_index=1)
    incoming["outputs"]["magnetic_check"]["final_frame_index"] = 0
    report = diagnose_result_magnetism(incoming, manager())["outputs"]["magnetic_check"]
    assert report["status"] == "unknown" and "final frame" in report["error"]


def test_parser_keeps_relax_frames_and_warning(tmp_path):
    import numpy as np
    from scientific_layer.dft.parse_vasp_result import parse_vasp_result
    struct = Structure.from_dict(result()["outputs"]["structure"])
    frames = [{"structure": struct, "e_0_energy": energy, "forces": np.zeros((3,3)),
               "electronic_steps": [{}]} for energy in (-8, -9)]
    parsed = SimpleNamespace(final_structure=struct, final_energy=-9, converged=True,
        converged_ionic=True, ionic_steps=frames, parameters={"NELM": 50})
    (tmp_path / "vasprun.xml").write_text("synthetic", encoding="utf-8")
    with patch("Process_Vasp.read_dft_magnetic_data", return_value={
            "status": "available", "elements": ["Fe", "Mn", "O"], "moments": [4.3, 1, 0]}):
        stored = parse_vasp_result(tmp_path, parser=lambda _: parsed)
    assert stored["status"] == "completed", stored
    assert stored["outputs"]["magnetic_moments"]["status"] == "available"
    assert stored["outputs"]["magnetic_moments"]["final_frame_index"] == 1
    assert "magnetic_check" not in stored["outputs"] and "checks_passed" not in stored
    assert len(stored["outputs"]["training_frames"]) == 2
    assert stored["outputs"]["energy"] == -9

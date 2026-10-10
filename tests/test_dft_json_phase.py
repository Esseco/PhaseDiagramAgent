from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pymatgen.core import Structure, Lattice

from phase_agent.analysis.phase.identify_result_phase import identify_result_phase
from phase_agent.analysis.phase.ensure_phase_identification import ensure_phase_identification
from phase_agent.science.structures.load_result_structure import load_result_structure
from phase_agent.analysis.phase.check_na_layer_uniformity import check_na_layer_uniformity


def json_result():
    structure = Structure(Lattice.cubic(4), ["Na", "Fe", "O", "O"],
                          [[0, 0, 0], [.5, .5, .5], [.25, .25, .25], [.75, .75, .75]])
    return {"task_id": "D1", "structure_id": "S1", "stage": "dft_single_point",
            "status": "completed", "model_version": "m1", "outputs": {
                "structure": structure.as_dict(), "energy": -10.,
                "final_frame_valid": True, "training_frame_index": 2, "final_frame_index": 2}}


def manager():
    return SimpleNamespace(data={"structures": {"S1": {"branch_id": "B1"}},
                                "branches": {"B1": {"P": "O3"}}}, boundary=None)


def test_json_phase_cache_tracks_content_not_file():
    result = json_result()
    cache = {}
    with patch("phase_agent.science.structures.identify_layered_phase_fast.identify_layered_phase_fast",
               return_value={"phase": "O3", "method": "test"}) as identify:
        first = identify_result_phase(result, manager(), cache=cache)
        second = identify_result_phase(result, manager(), cache=cache)
        assert first["actual_phase"] == second["actual_phase"] == "O3"
        assert identify.call_count == 1
        changed = deepcopy(result)
        changed["outputs"]["structure"]["lattice"]["matrix"][0][0] = 4.1
        identify_result_phase(changed, manager(), cache=cache)
        assert identify.call_count == 2


def test_invalid_last_frame_not_replaced_by_training_history():
    result = json_result()
    result["outputs"]["final_frame_valid"] = False
    with pytest.raises(ValueError, match="final frame"):
        load_result_structure(result["outputs"])
    result["outputs"]["final_frame_valid"] = True
    result["outputs"]["training_frame_index"] = 1
    with pytest.raises(ValueError, match="actual final frame"):
        load_result_structure(result["outputs"])


def test_json_phase_record_refresh_without_task_or_file():
    result = json_result()
    record = {"structure_id": "S1", "source_task_id": "D1", "energy_method": "dft",
              "energy": -10., "structure": result["outputs"]["structure"],
              "composition": {"Na": 1, "Fe": 1, "O": 2}, "final_frame_valid": True}
    with patch("phase_agent.science.structures.identify_layered_phase_fast.identify_layered_phase_fast",
               return_value={"phase": "O3", "method": "test"}):
        state, _ = ensure_phase_identification({"phase_records": [record]}, manager())
    assert state["phase_records"][0]["phase"] == "O3"
    assert state["phase_records"][0]["structure_sha256"]


def test_na_layer_check_accepts_json():
    result = json_result()
    with patch("Process_Vasp.structure.check_layer_equal", return_value=True) as check:
        actual = check_na_layer_uniformity(None, {"Na": 1}, structure_data=result["outputs"]["structure"])
    assert actual["status"] == "checked"
    assert isinstance(check.call_args.args[0], Structure)

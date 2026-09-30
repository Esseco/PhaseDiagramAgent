"""A recovered final structure is classified before entering a hull, once per content."""

from types import SimpleNamespace
from unittest.mock import patch

from pymatgen.core import Lattice, Structure

from analysis_layer.phase.identify_result_phase import identify_result_phase
from analysis_layer.phase.ensure_phase_identification import ensure_phase_identification
from analysis_layer.phase.update_phase_diagram import update_phase_diagram
from data_layer.ledger.collect_calculation_results import _phase_record


def test_same_structure_is_identified_once_and_changed_file_reidentified(tmp_path):
    path = tmp_path / "final.vasp"
    path.write_text("first")
    manager = SimpleNamespace(data={"branches": {"B": {"P": "O3"}},
                                    "structures": {}, "system_config": {}})
    result = {"stage": "relax_and_feature", "status": "completed", "branch_id": "B",
              "outputs": {"structure_path": str(path)}}
    structure = Structure(Lattice.cubic(4), ["Na", "O"], [[0, 0, 0], [.5, .5, .5]])
    cache = {}
    with patch("pymatgen.core.Structure.from_file", return_value=structure), patch(
            "scientific_layer.structures.identify_branch.identify_phase",
            return_value={"phase": "P3", "method": "test"}) as detector:
        first = identify_result_phase(result, manager, cache=cache)
        again = identify_result_phase(result, manager, cache=cache)
        assert detector.call_count == 1
        assert first["actual_phase"] == again["actual_phase"] == "P3"
        path.write_text("changed")
        identify_result_phase(result, manager, cache=cache)
        assert detector.call_count == 2


def test_unknown_phase_is_retried_and_identified_cache_survives_restart(tmp_path):
    path = tmp_path / "final.vasp"
    path.write_text("mock", encoding="utf-8")
    structure = Structure(Lattice.cubic(4), ["Na", "O"], [[0, 0, 0], [.5, .5, .5]])
    manager = SimpleNamespace(data={"branches": {"B": {"P": "O3"}},
                                    "structures": {"S": {"branch_id": "B"}},
                                    "system_config": {}})
    state = {"tasks": [{"task_id": "T1", "stage": "relax_and_feature",
                        "status": "completed", "branch_id": "B", "structure_id": "S",
                        "model_version": "m1", "outputs": {"structure_path": str(path),
                        "energy": -4.0}}],
             "phase_records": [{"record_id": "R1", "source_task_id": "T1",
                                "stage": "relax_and_feature", "structure_id": "S",
                                "model_version": "m1", "energy_method": "mlip",
                                "energy": -4.0, "structure_path": str(path),
                                "status": "pending_phase_identification"}]}
    cache_path = tmp_path / "phase_identification_cache.json"
    with patch("pymatgen.core.Structure.from_file", return_value=structure), patch(
            "scientific_layer.structures.identify_branch.identify_phase",
            side_effect=[{"phase": "X"}, {"phase": "P3", "method": "test"}]) as detector:
        first, _ = ensure_phase_identification(state, manager, cache_path=cache_path)
        assert first["phase_records"][0]["status"] == "pending_phase_identification"
        second, _ = ensure_phase_identification(first, manager, cache_path=cache_path)
        assert second["phase_records"][0]["phase"] == "P3"
        assert detector.call_count == 2
        resumed, _ = ensure_phase_identification(state, manager, cache_path=cache_path)
        assert resumed["phase_records"][0]["phase"] == "P3"
        assert detector.call_count == 2


def test_empty_na_endpoint_uses_mother_structure_and_supercell(tmp_path):
    path = tmp_path / "empty.vasp"
    path.write_text("mock")
    lattice = Lattice.cubic(4)
    empty = Structure(lattice, ["Fe", "O", "O"],
                      [[0, 0, 0], [.5, .5, .5], [.25, .25, .25]])
    full = Structure(lattice, ["Na", "Fe", "O", "O"],
                     [[.1, .1, .1], [0, 0, 0], [.5, .5, .5], [.25, .25, .25]])
    identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    boundary = {"P": {"at_x": {"0": ["O1"], "1": ["O3"]},
                      "intermediate": ["O3", "P3"]},
                "H": {phase: [identity] for phase in ("O1", "O3", "P3")}}
    manager = SimpleNamespace(boundary=boundary, data={
        "branches": {"B": {"P": "O1"}}, "structures": {}, "system_config": {}})
    result = {"stage": "relax_and_feature", "status": "completed", "branch_id": "B",
              "outputs": {"structure_path": str(path)}}
    with patch("pymatgen.core.Structure.from_file", return_value=empty):
        identified = identify_result_phase(result, manager,
            phase_references={"O1": full, "O3": full, "P3": full}, cache={})
    assert identified["actual_phase"] == "O1"
    assert identified["outputs"]["phase_identification"]["method"] == "reference_supercell"


def test_unidentified_result_remains_in_ledger_but_not_hull():
    manager = SimpleNamespace(data={"branches": {"B": {"P": "O3"}},
                                    "structures": {"S": {"branch_id": "B"}}})
    base = {"model_version": "m1", "converged": True, "outputs": {
        "energy": -4, "energy_unit": "eV", "composition": {"Na": 1, "Fe": 1, "O": 2}}}
    unknown = _phase_record(manager, "S", "R1", "relax_and_feature", base)
    assert unknown["status"] == "pending_phase_identification"
    assert unknown["phase"] is None
    identified = _phase_record(manager, "S", "R2", "relax_and_feature", {
        **base, "outputs": {**base["outputs"], "actual_phase": "P3",
                            "phase_identification": {"status": "identified", "phase": "P3"}}})
    diagram = update_phase_diagram([unknown, identified],
                                    active_model_version="m1")["diagrams"]["mlip"]
    assert diagram["entries"] == []
    assert diagram["status"] == "unknown"
    assert diagram["reason"] == "insufficient_Na_endpoints"

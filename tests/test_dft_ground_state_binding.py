from hashlib import sha256
from pymatgen.core import Lattice, Structure
import pytest
from phase_agent.analysis.phase.phase_dft_candidates import bind_ground_state_frames


def fixture(tmp_path):
    def make(name, shift):
        p = tmp_path / name
        Structure(
            Lattice.cubic(4),
            ["Na", "Ni", "O", "O"],
            [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75 + shift]],
        ).to(filename=str(p), fmt="poscar")
        return str(p), sha256(p.read_bytes()).hexdigest()

    old, oldsha = make("old.vasp", 0)
    new, newsha = make("new.vasp", 0.1)
    row = {
        "record_id": "gs1",
        "structure_id": "s1",
        "structure_path": old,
        "structure_sha256": oldsha,
        "is_stable": True,
        "phase": "O3",
        "phase_identification_status": "identified",
        "composition": {"Na": 1, "Ni": 1, "O": 2},
        "x_Na_per_O2": 1,
        "ehull": 0.0,
        "ehull_unit": "eV/atom",
        "energy_per_O2": -1.0,
    }
    state = {
        "active_model_version": "m1",
        "phase_diagrams": {
            "mlip": {
                "status": "completed",
                "model_version": "m1",
                "version": "h1",
                "entries": [row],
            }
        },
    }
    candidate = {
        "candidate_id": "s1",
        "structure_id": "s1",
        "branch_id": "b1",
        "structure_path": new,
        "structure_sha256": newsha,
        "predicted_Ehull": 0.2,
        "qbc": {"status": "completed", "f_std_max": 1},
    }
    return state, candidate, row


def test_binds_gs_frame_and_does_not_borrow_qbc(tmp_path):
    state, candidate, row = fixture(tmp_path)
    result = bind_ground_state_frames(state, [candidate])[0]
    assert result["structure_path"] == row["structure_path"]
    assert result["structure_sha256"] == row["structure_sha256"]
    assert result["predicted_Ehull"] == 0
    assert result["qbc"]["status"] == "not_available_exact_current_geometry"
    assert result["branch_id"] == "b1"


def test_inequivalent_gs_same_id_is_rejected(tmp_path):
    state, candidate, row = fixture(tmp_path)
    other = dict(
        row,
        record_id="gs2",
        structure_path=candidate["structure_path"],
        structure_sha256=candidate["structure_sha256"],
    )
    state["phase_diagrams"]["mlip"]["entries"].append(other)
    with pytest.raises(ValueError, match="ground_state_inequivalent"):
        bind_ground_state_frames(state, [candidate])


def test_corrupt_gs_is_rejected(tmp_path):
    state, candidate, row = fixture(tmp_path)
    row["structure_sha256"] = "bad"
    with pytest.raises(ValueError, match="ground_state_frame_invalid"):
        bind_ground_state_frames(state, [candidate])


def test_selection_requires_ground_state_and_actual_frame(tmp_path):
    from phase_agent.tools.workflows.validate_dft_agent_decisions import (
        _ground_state_selection_errors,
    )

    state, candidate, row = fixture(tmp_path)
    state["qbc_candidates"] = [candidate]
    assert _ground_state_selection_errors(state, {"other"}) == [
        "ground_state_selection_required:s1"
    ]
    assert _ground_state_selection_errors(state, {"s1"}) == [
        "ground_state_frame_binding_required:s1"
    ]
    state["qbc_candidates"] = bind_ground_state_frames(state, [candidate])
    assert _ground_state_selection_errors(state, {"s1"}) == []

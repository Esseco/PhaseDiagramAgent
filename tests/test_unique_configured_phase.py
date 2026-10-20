from types import SimpleNamespace
from unittest.mock import patch
import pytest
from pymatgen.core import Lattice, Structure
from phase_agent.analysis.phase.identify_result_phase import identify_result_phase


@pytest.mark.parametrize("x,phase", [(0, "O1"), (1, "O3"), (0.5, "P2")])
def test_final_composition_unique_phase_bypasses_all_detectors(tmp_path, x, phase):
    species = ["Fe", "Fe", "O", "O", "O", "O"] + ["Na"] * int(2 * x)
    path = tmp_path / "s.vasp"
    Structure(
        Lattice.cubic(6), species, [[i * 0.1, 0, 0] for i in range(len(species))]
    ).to(filename=path, fmt="poscar")
    manager = SimpleNamespace(
        boundary={"P": {"at_x": {"0": ["O1"], "1": ["O3"]}, "intermediate": ["P2"]}},
        data={"branches": {"B": {"P": "P3"}}, "system_config": {}},
    )
    task = {
        "stage": "relax_and_feature",
        "status": "completed",
        "branch_id": "B",
        "outputs": {"structure_path": str(path), "energy": -5},
    }
    with (
        patch(
            "phase_agent.science.structures.identify_branch.identify_phase",
            side_effect=AssertionError("no detector"),
        ),
        patch(
            "phase_agent.science.structures.identify_branch.identify_supercell",
            side_effect=AssertionError("no matcher"),
        ),
        patch(
            "phase_agent.science.structures.identify_layered_phase_fast.identify_layered_phase_fast",
            side_effect=AssertionError("no fast detector"),
        ),
    ):
        result = identify_result_phase(task, manager)
    assert result["actual_phase"] == phase
    assert result["source_phase"] == "P3"
    assert result["outputs"]["energy"] == -5
    assert (
        result["outputs"]["phase_identification"]["method"] == "unique_configured_phase"
    )


def test_multiple_phases_still_require_identification(tmp_path):
    path = tmp_path / "s.vasp"
    Structure(
        Lattice.cubic(6),
        ["Na", "Fe", "O", "O"],
        [[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0], [0.3, 0, 0]],
    ).to(filename=path, fmt="poscar")
    manager = SimpleNamespace(
        boundary={"P": ["O3", "P3"]}, data={"branches": {}, "system_config": {}}
    )
    task = {
        "stage": "relax_and_feature",
        "status": "completed",
        "outputs": {"structure_path": str(path)},
    }
    with patch(
        "phase_agent.science.structures.identify_layered_phase_fast.identify_layered_phase_fast",
        return_value={"phase": "P3", "method": "mock"},
    ) as detect:
        result = identify_result_phase(task, manager)
    detect.assert_called_once()
    assert result["actual_phase"] == "P3"


def test_missing_final_structure_is_not_invented_from_configuration():
    manager = SimpleNamespace(
        boundary={"P": ["O3"]}, data={"branches": {}, "system_config": {}}
    )
    result = identify_result_phase(
        {"stage": "relax_and_feature", "status": "completed", "outputs": {}}, manager
    )
    assert result["phase_identification_status"] == "unknown"


def test_completed_refresh_does_not_claim_waiting_for_upload():
    from phase_agent.runtime.chat_state_presentation import brief_chat_state

    state = {
        "model_refresh": {"status": "waiting_results", "task_ids": ["T"]},
        "tasks": [
            {
                "task_id": "T",
                "status": "completed",
                "phase_identification_status": "unknown",
            }
        ],
    }
    text = brief_chat_state(state)
    assert "全部回收" in text and "相识别受阻" in text
    assert "等待结果" not in text

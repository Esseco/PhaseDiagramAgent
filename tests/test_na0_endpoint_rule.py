from types import SimpleNamespace
from unittest.mock import patch
from pymatgen.core import Structure, Lattice
from phase_agent.analysis.phase.identify_result_phase import (
    identify_result_phase,
    _classification_scope,
)


def setup_result(tmp_path, phases):
    path = tmp_path / "zero.vasp"
    Structure(
        Lattice.cubic(5),
        ["Fe", "O", "O"],
        [[0, 0, 0], [0.2, 0.2, 0.2], [0.4, 0.4, 0.4]],
    ).to(filename=path, fmt="poscar")
    manager = SimpleNamespace(
        data={"branches": {"B": {"P": "O3"}}, "structures": {}, "system_config": {}},
        boundary={"P": phases},
    )
    result = {
        "stage": "relax_and_feature",
        "status": "completed",
        "branch_id": "B",
        "outputs": {"structure_path": str(path), "energy": -4},
    }
    return manager, result


def test_na0_uses_confirmed_o1_rule_without_geometry_match(tmp_path):
    manager, result = setup_result(
        tmp_path, {"at_x": {"0": ["O1"], "1": ["O3"]}, "intermediate": ["O3", "P3"]}
    )
    with patch(
        "phase_agent.science.structures.identify_branch.identify_supercell",
        side_effect=AssertionError("must not match geometry"),
    ):
        identified = identify_result_phase(result, manager)
    assert identified["actual_phase"] == "O1"
    assert (
        identified["outputs"]["phase_identification"]["method"]
        == "unique_configured_phase"
    )
    assert identified["outputs"]["energy"] == -4


def test_na0_does_not_add_o1_to_single_phase_cr_boundary(tmp_path):
    manager, result = setup_result(tmp_path, ["O3"])
    with patch(
        "phase_agent.science.structures.identify_branch.identify_supercell",
        return_value=("O3", None, None),
    ) as match:
        identified = identify_result_phase(
            result,
            manager,
            phase_references={"O3": result["outputs"]["structure_path"]},
        )
    assert identified["actual_phase"] == "O3"
    match.assert_not_called()


def test_only_na0_cache_scope_changes_and_old_o3_evidence_is_replaced(tmp_path):
    manager, result = setup_result(
        tmp_path, {"at_x": {"0": ["O1"], "1": ["O3"]}, "intermediate": ["O3", "P3"]}
    )
    old = _classification_scope(manager, {})
    new = _classification_scope(manager, {}, configured_phase="O1")
    assert old != new
    first = identify_result_phase(result, manager)
    result["outputs"]["phase_identification"] = {
        **first["outputs"]["phase_identification"],
        "classification_scope": old,
        "phase": "O3",
    }
    assert identify_result_phase(result, manager)["actual_phase"] == "O1"


def test_supplement_scope_distinguishes_two_geometries_of_same_structure():
    from phase_agent.tools.workflows.model_refresh_scope import refresh_scope_errors
    from copy import deepcopy

    refresh = {
        "new_model_version": "new",
        "wave": 1,
        "plan": {
            "checksum": "p",
            "candidates": [
                {"structure_id": "S", "structure_sha256": sha, "operation": "predict"}
                for sha in ("a", "b")
            ],
        },
        "approved_scope": {
            "plan_checksum": "p",
            "model_version": "new",
            "config_version": "c",
            "remaining_waves": 1,
            "eligible_structure_ids": ["S", "S"],
            "maximum_count": 2,
            "maximum_cost": 2,
        },
        "supplemental_candidates": [
            {
                "structure_id": "S",
                "structure_sha256": sha + "-result",
                "approved_input_structure_sha256": sha,
                "relative_cost": 1,
            }
            for sha in ("a", "b")
        ],
    }
    assert refresh_scope_errors(refresh, "c") == []
    repeated = deepcopy(refresh)
    repeated["supplemental_candidates"][1] = deepcopy(
        repeated["supplemental_candidates"][0]
    )
    assert refresh_scope_errors(repeated, "c")
    outside = deepcopy(refresh)
    outside["supplemental_candidates"][1]["approved_input_structure_sha256"] = (
        "unapproved"
    )
    assert refresh_scope_errors(outside, "c")

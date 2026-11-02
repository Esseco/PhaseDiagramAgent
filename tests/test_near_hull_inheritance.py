import hashlib

import pytest
from pymatgen.core import Lattice, Structure

from phase_agent.science.structures.inherit_near_hull_occupancy import inherited_na_structures


def test_adjacent_na_preserves_parent_frame_and_checks_checksum(tmp_path, monkeypatch):
    from phase_agent.science.structures import inherit_near_hull_occupancy as module

    parent = Structure(
        Lattice.cubic(10),
        ["Na", "Ni", "Ni", "O", "O", "O", "O"],
        [
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
            [0.7, 0.7, 0.7],
            [0.3, 0.3, 0.3],
            [0.4, 0.4, 0.4],
            [0.6, 0.6, 0.6],
            [0.8, 0.8, 0.8],
        ],
    )
    reference = parent.copy()
    reference.append("Na", [0.9, 0.9, 0.9])
    file = tmp_path / "parent.vasp"
    parent.to(filename=str(file), fmt="poscar")
    checksum = hashlib.sha256(file.read_bytes()).hexdigest()
    h = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    source = {
        "phase": "O3",
        "branch_H": h,
        "branch_x": "1/2",
        "structure_path": str(file),
        "structure_sha256": checksum,
    }
    branch = {"P": "O3", "H": h, "x": "1", "T": ["Ni", "Ni"], "near_hull_parent_evidence": source}
    created = inherited_na_structures(branch, reference, {"TM_ratio": {"Ni": 1}}, {}, 3, 1)
    assert len(created) == 1
    child, full, vector = created[0]
    assert child.composition["Na"] == 2
    assert vector == [1, 1]
    assert [list(site.frac_coords) for site in child if site.specie.symbol == "Ni"] == [
        list(site.frac_coords) for site in parent if site.specie.symbol == "Ni"
    ]
    assert full.composition["Na"] == 2
    branch["x"] = "0"
    child, _, vector = inherited_na_structures(
        branch, reference, {"TM_ratio": {"Ni": 1}}, {}, 3, 1
    )[0]
    assert child.composition.get("Na", 0) == 0
    assert vector == [0, 0]
    source["structure_sha256"] = "wrong"
    with pytest.raises(ValueError, match="checksum"):
        inherited_na_structures(branch, reference, {"TM_ratio": {"Ni": 1}}, {}, 3, 1)


def test_initialization_uses_all_reference_na_sites_and_parent_based_result(monkeypatch):
    from phase_agent.science.structures import initialize_branch_structures as module
    from phase_agent.science.structures import inherit_near_hull_occupancy as inherited_module

    reference = Structure(
        Lattice.cubic(10),
        ["Na", "Na", "Na", "Ni", "Ni", "O", "O", "O", "O"],
        [
            [0.1, 0.1, 0.1],
            [0.5, 0.5, 0.5],
            [0.9, 0.9, 0.9],
            [0.2, 0.2, 0.2],
            [0.7, 0.7, 0.7],
            [0.3, 0.3, 0.3],
            [0.4, 0.4, 0.4],
            [0.6, 0.6, 0.6],
            [0.8, 0.8, 0.8],
        ],
    )
    stoichiometric = reference.copy()
    stoichiometric.remove_sites([2])
    seen = []

    def unexpected_regeneration(*args, **kwargs):
        raise AssertionError("composition must not regenerate an unrelated parent frame")

    monkeypatch.setattr(module, "generate_branch_structure", unexpected_regeneration)

    def inherit(branch, full, boundary, refs, limit, seed):
        seen.append(full.composition["Na"])
        return [(stoichiometric.copy(), reference.copy(), [1, 1, 0])]

    monkeypatch.setattr(inherited_module, "inherited_na_structures", inherit)
    h = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    branch = {
        "P": "O3",
        "H": h,
        "x": "1",
        "T": ["Ni", "Ni"],
        "strategy": "composition",
        "near_hull_parent_evidence": {"record_id": "r"},
    }
    result = module.initialize_branch_structures(
        [branch],
        {"TM_ratio": {"Ni": 1}},
        {"O3": reference},
        initial_states_per_branch=3,
        seed=4,
    )
    assert seen == [3]
    assert result[0]["V"] == [1, 1, 0]
    assert result[0]["initialization_method"] == "near_hull_parent_adjacent_na"

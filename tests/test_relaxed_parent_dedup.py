import hashlib
from types import SimpleNamespace

import pytest
from pymatgen.core import Lattice, Structure

from phase_agent.science.structures.deduplicate_candidates import deduplicate_candidates
from phase_agent.tools.workflows import region_controller
from phase_agent.science.structures.inherit_near_hull_occupancy import inherited_na_structures


def make_case(tmp_path):
    h = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    ref = Structure(
        Lattice.cubic(5),
        ["Na", "Ni", "O", "O"],
        [[0, 0, 0], [0.2, 0.2, 0.2], [0.4, 0.4, 0.4], [0.6, 0.6, 0.6]],
    )
    parent = ref.copy()
    parent.lattice = Lattice.cubic(7)
    path = tmp_path / "parent.vasp"
    parent.to(filename=str(path), fmt="poscar")
    source = {
        "branch_id": "parent",
        "phase": "O3",
        "branch_H": h,
        "branch_x": "1",
        "structure_path": str(path),
        "structure_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    record = {"P": "O3", "H": h, "x": "1", "T": ["Ni"], "composition": {"O": 2}}
    child = parent.copy()
    child.remove_species(["Na"])
    candidate = {
        "P": "O3",
        "H": h,
        "x": "0",
        "T": ["Ni"],
        "structure": child,
        "strategy": "composition",
        "parent_branch_id": "parent",
        "near_hull_parent_evidence": source,
        "initialization_method": "near_hull_parent_adjacent_na",
        "arrangement": {"V": [0]},
    }
    manager = SimpleNamespace(
        boundary={"P": ["O3"], "H": [h], "TM_ratio": {"Ni": 1}},
        data={"branches": {"parent": record}, "structures": {}},
    )
    return manager, {"O3": ref}, candidate


def test_strained_inherited_parent_uses_verified_branch_identity(tmp_path):
    manager, refs, candidate = make_case(tmp_path)
    result = deduplicate_candidates(
        [candidate, dict(candidate)], manager=manager, phase_references=refs
    )
    assert result["statistics"]["unique"] == 1
    assert result["statistics"]["configuration_duplicates"] == 1
    assert result["unique_candidates"][0]["H"] == candidate["H"]


@pytest.mark.parametrize(
    "tamper", ["parent_id", "H", "lattice", "framework", "checksum", "boundary"]
)
def test_inherited_parent_provenance_is_checked(tmp_path, tamper):
    manager, refs, candidate = make_case(tmp_path)
    if tamper == "parent_id":
        candidate["parent_branch_id"] = "other"
    if tamper == "H":
        candidate["H"] = [[2, 0, 0], [0, 1, 0], [0, 0, 1]]
    if tamper == "lattice":
        candidate["structure"].lattice = Lattice.cubic(8)
    if tamper == "framework":
        candidate["structure"].translate_sites([0], [0.1, 0, 0])
    if tamper == "checksum":
        candidate["near_hull_parent_evidence"]["structure_sha256"] = "bad"
    if tamper == "boundary":
        manager.boundary["P"] = {"at_x": {"0": ["O1"]}, "intermediate": ["O3"]}
    with pytest.raises((ValueError, KeyError)):
        deduplicate_candidates([candidate], manager=manager, phase_references=refs)


def test_single_tm_sampling_needs_no_geometric_diversity(monkeypatch):
    class Manager:
        boundary = {"TM_ratio": {"Ni": 1}}
        data = {
            "branches": {
                "old": {"P": "O3", "H": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "x": "1", "T": ["Ni"]}
            },
            "structures": {},
        }

        def add_branch(self, **kwargs):
            self.data["branches"]["new"] = kwargs

    monkeypatch.setattr(
        region_controller, "Diversity", lambda *_: pytest.fail("unnecessary geometric diversity")
    )
    params = {
        "sampling_mode": "region_based",
        "regions": [
            {
                "region_id": "r",
                "priority": 1,
                "sample_count": 1,
                "strategy": "coverage",
                "phase": "O3",
                "min_det_H": 1,
                "max_det_H": 1,
                "na_min": 0,
                "na_max": 1,
                "reason": "coverage",
                "evidence_refs": [],
            }
        ],
    }
    from phase_agent.policies.region_sampling import MODE

    params["sampling_mode"] = MODE
    candidate = {"P": "O3", "H": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "x": "0", "T": ["Ni"]}
    chosen, _ = region_controller.sample_regions(
        Manager(), {}, params, producer=lambda *a, **k: [candidate]
    )
    assert len(chosen) == 1


def test_inherited_occupancy_vector_uses_reference_site_order(tmp_path):
    h = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    ref = Structure(
        Lattice.cubic(10),
        ["Na", "Na", "Na", "Ni", "O", "O"],
        [
            [0.1, 0.1, 0.1],
            [0.5, 0.5, 0.5],
            [0.9, 0.9, 0.9],
            [0.2, 0.2, 0.2],
            [0.3, 0.3, 0.3],
            [0.7, 0.7, 0.7],
        ],
    )
    parent = ref.copy()
    parent.remove_sites([0, 2])
    path = tmp_path / "parent.vasp"
    parent.to(filename=str(path), fmt="poscar")
    source = {
        "phase": "O3",
        "branch_H": h,
        "branch_x": "1",
        "structure_path": str(path),
        "structure_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    branch = {"P": "O3", "H": h, "x": "0", "T": ["Ni"], "near_hull_parent_evidence": source}
    child, full, vector = inherited_na_structures(branch, ref, {"TM_ratio": {"Ni": 1}}, {}, 1, 1)[0]
    assert vector == [0, 0, 0]
    na = [list(s.frac_coords) for s in full if s.specie.symbol == "Na"]
    assert na == [[0.1, 0.1, 0.1], [0.5, 0.5, 0.5], [0.9, 0.9, 0.9]]

import unittest
from collections import Counter

from pymatgen.core import Lattice, Structure

from scientific_layer.structures.enumerate_legal_frameworks import enumerate_legal_frameworks
from scientific_layer.structures.propose_branches import propose_branches
from data_layer.ledger.phase_data_manager import PhaseDataManager
from scientific_layer.structures.generate_branch_structure import generate_branch_structure
from scientific_layer.structures.generate_tm_ordering_branches import generate_tm_ordering_branches
from scientific_layer.structures.boundary_utils import det_H
from config_layer.defaults.layered_oxide_system_config import layered_oxide_system_config


def build_case():
    h1 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    h2 = [[2, 0, 0], [0, 1, 0], [0, 0, 1]]
    boundary = {
        "P": ["O3", "P3"],
        "H": [h1, h2],
        "TM_ratio": {"Fe": 1, "Mn": 1},
    }
    species = ["Na", "Co", "Co", "O", "O"]
    coords = [[0, 0, 0.1], [0, 0, 0.4], [0.5, 0.5, 0.6], [0, 0.5, 0.3], [0.5, 0, 0.7]]
    references = {
        "O3": Structure(Lattice.hexagonal(3, 10), species, coords),
        "P3": Structure(Lattice.hexagonal(4, 12), species, coords),
    }
    manager = PhaseDataManager(boundary)
    parent_structure = generate_branch_structure(
        boundary,
        phase="O3",
        H=h1,
        x=1,
        T=["Fe", "Mn"],
        phase_references=references,
        seed=3,
    )
    parent = manager.identify_branch(
        parent_structure,
        phase_references=references,
        use_coordination_phase=False,
    )["branch_id"]
    mappings = {
        (f"O3:{h1}", f"P3:{h1}"): [0, 1],
        (f"O3:{h1}", f"O3:{h2}"): [0, 1, 0, 1],
    }
    return manager, references, parent, mappings


class BranchProposalTest(unittest.TestCase):
    def test_fixed_tm_uses_reference_order_and_skips_tm_mutation(self) -> None:
        manager, references, _, _ = build_case()
        for reference in references.values():
            reference.replace(1, "Fe")
            reference.replace(2, "Mn")
        system = layered_oxide_system_config(boundary=manager.boundary)
        system["configuration_space"]["roles"]["T"] = "fixed"
        system["configuration_space"]["fixed_T_source"] = "phase_reference"
        manager = PhaseDataManager(manager.boundary, system_config=system)
        first = propose_branches(manager, references,
                                 quotas={"coverage": 4, "tm_ordering": 4}, seed=1,
                                 register=False)
        second = propose_branches(manager, references,
                                  quotas={"coverage": 4, "tm_ordering": 4}, seed=99,
                                  register=False)
        self.assertTrue(first)
        self.assertTrue(all(item["strategy"] != "tm_ordering" for item in first))
        self.assertEqual([item["T"] for item in first], [item["T"] for item in second])

    def test_generation_cap_filters_final_candidates(self) -> None:
        manager, references, parent, mappings = build_case()
        candidates = propose_branches(
            manager, references, quotas={"coverage": 10}, seed=7,
            parent_branch_ids=[parent], site_mappings=mappings,
            max_det_H=1,
        )
        self.assertTrue(candidates)
        self.assertTrue(all(det_H(item["H"]) <= 1 for item in candidates))

    def test_coverage_can_revisit_region_with_distinct_tm_orders(self) -> None:
        manager, references, _, _ = build_case()
        candidates = propose_branches(
            manager, references, quotas={"coverage": 20}, seed=17,
            max_det_H=1, register=False,
        )
        self.assertEqual(len(candidates), 20)
        self.assertTrue(all(det_H(item["H"]) == 1 for item in candidates))
        self.assertTrue(all(item["branch_id"] is None for item in candidates))

    def test_framework_enumeration(self) -> None:
        manager, references, _, _ = build_case()
        frameworks = enumerate_legal_frameworks(manager.boundary, references)
        self.assertEqual(len(frameworks), 4)
        self.assertEqual(frameworks[0]["allowed_x"], ["0", "1"])

    def test_all_strategies_and_reproducibility(self) -> None:
        quotas = {
            "coverage": 1,
            "composition": 1,
            "competing_phase": 1,
            "tm_ordering": 1,
            "periodic_extension": 1,
        }
        first_manager, first_refs, first_parent, first_maps = build_case()
        first = propose_branches(
            first_manager,
            first_refs,
            quotas=quotas,
            seed=9,
            parent_branch_ids=[first_parent],
            site_mappings=first_maps,
        )
        second_manager, second_refs, second_parent, second_maps = build_case()
        second = propose_branches(
            second_manager,
            second_refs,
            quotas=quotas,
            seed=9,
            parent_branch_ids=[second_parent],
            site_mappings=second_maps,
        )
        fields = (
            "P",
            "H",
            "x",
            "T",
            "strategy",
            "parent_branch_id",
            "seed",
            "inheritance",
            "branch_id",
        )
        self.assertEqual(
            [{key: item[key] for key in fields} for item in first],
            [{key: item[key] for key in fields} for item in second],
        )
        self.assertEqual({item["strategy"] for item in first}, set(quotas))
        self.assertTrue(all(item["structure"] is not None for item in first))
        ordered = next(item for item in first if item["strategy"] == "tm_ordering")
        parent_T = first_manager.data["branches"][first_parent]["T"]
        self.assertEqual(Counter(ordered["T"]), Counter(parent_T))
        self.assertNotEqual(ordered["T"], parent_T)

    def test_tm_ordering_rejects_parent_outside_frozen_composition(self) -> None:
        manager, references, parent, _ = build_case()
        manager.data["branches"][parent]["T"] = ["Fe", "Fe"]
        with self.assertRaisesRegex(ValueError, "冻结"):
            generate_tm_ordering_branches(
                manager,
                references,
                parent_branch_ids=[parent],
                quota=1,
                seed=9,
            )

    def test_missing_mapping_is_recorded(self) -> None:
        manager, references, parent, _ = build_case()
        candidates = propose_branches(
            manager,
            references,
            quotas={"competing_phase": 1},
            seed=4,
            parent_branch_ids=[parent],
        )
        self.assertEqual(candidates[0]["inheritance"], "regenerated_no_mapping")


if __name__ == "__main__":
    unittest.main()

"""初始化与登记流程的小样例。"""

from __future__ import annotations

import tempfile
import unittest
import random
from pathlib import Path
from unittest.mock import patch

from pymatgen.core import Lattice, Structure

from scientific_layer.structures.initialize_branch_structures import initialize_branch_structures
from data_layer.ledger.register_candidate_batch import register_candidate_batch
from scientific_layer.structures.identify_branch import extract_T
from data_layer.ledger.phase_data_manager import PhaseDataManager


class BranchGenerationFlowTest(unittest.TestCase):
    def setUp(self):
        self.H = [[2, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.reference = Structure(
            Lattice.hexagonal(3, 12),
            ["Na", "Fe", "O", "O"],
            [[0, 0, 0], [0, 0, 0.25], [0, 0, 0.4], [0, 0, 0.6]],
        )
        self.boundary = {
            "P": ["O3"],
            "H": {"O3": [self.H]},
            "TM_ratio": {"Fe": 1},
        }
        self.branch = {"P": "O3", "H": self.H, "x": "1/2", "T": ["Fe", "Fe"]}

    def initialize(self):
        return initialize_branch_structures(
            [self.branch],
            self.boundary,
            {"O3": self.reference},
            initial_states_per_branch=2,
            seed=17,
        )

    def test_seed_reproducibility_and_T_is_fixed(self):
        first, second = self.initialize(), self.initialize()
        self.assertEqual([item["V"] for item in first], [item["V"] for item in second])
        for item in first:
            T, _ = extract_T(item["structure"], self.boundary["TM_ratio"])
            self.assertEqual(T, self.branch["T"])
            self.assertTrue(item["structure"].is_ordered)
            self.assertIsInstance(item["electrostatic_energy"], float)
            self.assertIn(item["electrostatic_rank"], range(10))
            self.assertIsNotNone(item["full_na_structure"])
            self.assertEqual(item["full_na_structure"].composition["Na"], 2)

    def test_top_ten_selects_at_most_three(self):
        states = initialize_branch_structures(
            [self.branch], self.boundary, {"O3": self.reference},
            initial_states_per_branch=5, seed=17,
        )
        self.assertLessEqual(len(states), 3)

    def test_fixed_charge_retry_when_charge_balance_fails(self):
        with patch("scientific_layer.structures.rank_na_orderings_by_electrostatics."
                   "_assign_charge_balanced_average_tm_valence",
                   side_effect=ValueError("simulated charge assignment failure")):
            states = self.initialize()
        self.assertTrue(states)
        self.assertTrue(all(item["electrostatic_charge_scheme"] ==
                            "fallback_Na1_O-2_Fe3_Mn4" for item in states))

    def test_each_na_layer_must_be_occupied(self):
        reference = Structure(
            Lattice.hexagonal(3, 12),
            ["Na", "Na", "Fe", "Fe", "O", "O", "O", "O"],
            [[0, 0, 0], [0, 0, .5], [0, 0, .25], [0, 0, .75],
             [.1, 0, .1], [.2, 0, .2], [.1, 0, .6], [.2, 0, .7]],
        )
        h = [[2, 0, 0], [0, 1, 0], [0, 0, 1]]
        boundary = {"P": ["O3"], "H": {"O3": [h]}, "TM_ratio": {"Fe": 1}}
        branch = {"P": "O3", "H": h, "x": "1/2", "T": ["Fe"] * 4}
        states = initialize_branch_structures(
            [branch], boundary, {"O3": reference},
            initial_states_per_branch=5, seed=4,
        )
        self.assertTrue(states)
        self.assertEqual([state["electrostatic_rank"] for state in states],
                         random.Random(4).sample(range(4), 3))
        for state in states:
            occupied_z = [site.frac_coords[2] for site in state["structure"]
                          if site.is_ordered and site.specie.symbol == "Na"]
            self.assertTrue(any(abs(z % 1) < .02 for z in occupied_z))
            self.assertTrue(any(abs((z % 1) - .5) < .02 for z in occupied_z))
        impossible = {**branch, "x": "1/4"}
        with self.assertRaisesRegex(RuntimeError, "少于 2 个 Na 层"):
            initialize_branch_structures(
                [impossible], boundary, {"O3": reference},
                initial_states_per_branch=3, seed=4,
            )

    def test_na_zero_reference_without_na_sites(self):
        reference = Structure(
            Lattice.hexagonal(3, 12), ["Fe", "O", "O"],
            [[0, 0, .25], [0, 0, .4], [0, 0, .6]],
        )
        boundary = {"P": ["O1"], "H": {"O1": [self.H]}, "TM_ratio": {"Fe": 1}}
        branch = {"P": "O1", "H": self.H, "x": "0", "T": ["Fe", "Fe"]}
        states = initialize_branch_structures(
            [branch], boundary, {"O1": reference},
            initial_states_per_branch=3, seed=1,
        )
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0]["V"], [])
        self.assertIsNone(states[0]["full_na_structure"])

    def test_repeat_registration_reuses_ids(self):
        manager = PhaseDataManager(self.boundary)
        candidates = self.initialize()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            first = register_candidate_batch(
                manager, candidates, structure_directory=directory
            )
            second = register_candidate_batch(
                manager, candidates, structure_directory=directory
            )
            self.assertEqual(first, second)
            self.assertEqual(len(manager.data["branches"]), 1)
            self.assertEqual(len(manager.data["structures"]), 2)
            self.assertTrue(all(isinstance(row["metadata"].get("electrostatic_energy"), float)
                                for row in manager.data["structures"].values()))
            self.assertTrue(all(row["metadata"].get("electrostatic_charge_scheme")
                                == "charge_balanced"
                                for row in manager.data["structures"].values()))
            self.assertTrue(
                all(Path(item["structure_path"]).is_file() for item in first)
            )
            self.assertTrue(
                all(Path(item["full_na_structure_path"]).is_file() for item in first)
            )


if __name__ == "__main__":
    unittest.main()

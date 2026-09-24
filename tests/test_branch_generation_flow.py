"""初始化与登记流程的小样例。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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
            self.assertTrue(
                all(Path(item["structure_path"]).is_file() for item in first)
            )


if __name__ == "__main__":
    unittest.main()

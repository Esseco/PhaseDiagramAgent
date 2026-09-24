"""三个候选处理函数的小样例。"""

from __future__ import annotations

import unittest

from pymatgen.core import Lattice, Structure

from scientific_layer.structures.deduplicate_candidates import deduplicate_candidates
from execution_layer.budget.estimate_candidate_cost import estimate_candidate_cost
from scientific_layer.structures.select_candidates import select_candidates


class CandidateDecisionTest(unittest.TestCase):
    def setUp(self):
        self.H = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.structure = Structure(
            Lattice.hexagonal(3, 12),
            ["Na", "Fe", "Mn", "O", "O"],
            [[0, 0, 0], [0, 0, 0.25], [0.5, 0.5, 0.25], [0, 0, 0.4], [0, 0, 0.6]],
        )

    def candidate(self, identifier, *, x="1", parent=None):
        return {
            "candidate_id": identifier,
            "P": "O3",
            "H": self.H,
            "x": x,
            "T": ["Fe", "Mn"],
            "structure": self.structure.copy(),
            "strategy": "coverage",
            "parent_branch_id": parent,
            "seed": 7,
        }

    def test_deduplicate_keeps_sources(self):
        result = deduplicate_candidates([self.candidate("a"), self.candidate("b")])
        self.assertEqual(result["statistics"]["unique"], 1)
        self.assertEqual(result["statistics"]["configuration_duplicates"], 1)
        self.assertEqual(len(result["unique_candidates"][0]["generation_sources"]), 2)

    def test_cost_is_relative_and_has_no_wall_time(self):
        result = estimate_candidate_cost(
            [{**self.candidate("a"), "initial_state_count": 3}],
            config={"reference_atoms": 5, "planned_search_budget": 2},
        )[0]
        self.assertEqual(result["estimated_cost"]["value"], 6)
        self.assertIsNone(result["estimated_cost"]["wall_time"])

    def test_selection_is_reproducible_and_respects_budget(self):
        candidates = estimate_candidate_cost(
            [self.candidate("a"), self.candidate("b", x="1/2", parent="p")],
            config={"reference_atoms": 5},
        )
        options = {
            "batch_size": 2,
            "cost_budget": 1,
            "random_fraction": 0.5,
            "seed": 11,
        }
        first = select_candidates(candidates, **options)
        second = select_candidates(candidates, **options)
        self.assertEqual(
            [item["candidate_id"] for item in first["selected_candidates"]],
            [item["candidate_id"] for item in second["selected_candidates"]],
        )
        self.assertEqual(first["summary"]["selected"], 1)


if __name__ == "__main__":
    unittest.main()

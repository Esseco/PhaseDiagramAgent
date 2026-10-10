import unittest
from unittest.mock import patch

import numpy as np
from pymatgen.core import Lattice, Structure

from phase_agent.science.structures.enumerate_layered_oxide_supercells import (
    LAYERED_OXIDE_P_SMALL_RECOMMENDATIONS,
    can_contain,
    enumerate_layered_oxide_supercells,
)


class FakeSupercell:
    def __init__(self, cell):
        self.cell = cell


class LayeredOxideSupercellTest(unittest.TestCase):
    def setUp(self):
        self.structure = Structure(
            Lattice.from_parameters(3, 3, 20, 90, 90, 90),
            ["Li", "Co", "O", "O"],
            [[0, 0, 0.1], [0, 0, 0.4], [0.5, 0.5, 0.3], [0.5, 0, 0.7]],
        )

    def test_selected_matrix_contains_supercell_and_returns_project_format(self):
        h = np.diag([2, 2, 1])
        cell = h @ np.asarray(self.structure.lattice.matrix)
        with patch(
            "phase_agent.science.structures.enumerate_layered_oxide_supercells._enumerate_icet_supercells",
            return_value=[FakeSupercell(cell)],
        ):
            result = enumerate_layered_oxide_supercells(
                "O3",
                self.structure,
                sizes=[4],
                p_small_list=[LAYERED_OXIDE_P_SMALL_RECOMMENDATIONS[0]],
                min_distance=1.0,
            )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["phase"], "O3")
        self.assertEqual(result[0]["H"], [[2, 0, 0], [0, 2, 0], [0, 0, 1]])
        self.assertEqual(result[0]["det_H"], 4)
        self.assertEqual(result[0]["x_list"], ["1/4", "1/2", "3/4", "1"])
        self.assertEqual(result[0]["R_cut"], 3.0)

    def test_all_selected_p_small_matrices_are_required(self):
        h = np.diag([2, 2, 1])
        self.assertTrue(can_contain(h, [LAYERED_OXIDE_P_SMALL_RECOMMENDATIONS[0]]))
        self.assertFalse(can_contain(h, [LAYERED_OXIDE_P_SMALL_RECOMMENDATIONS[1]]))
        self.assertFalse(
            can_contain(h, LAYERED_OXIDE_P_SMALL_RECOMMENDATIONS)
        )
        contains_both = [[1, 2, 0], [0, 6, 0], [0, 0, 1]]
        self.assertTrue(
            can_contain(contains_both, LAYERED_OXIDE_P_SMALL_RECOMMENDATIONS)
        )

    def test_minimum_periodic_distance_filter_is_preserved(self):
        h = np.diag([2, 2, 1])
        cell = h @ np.asarray(self.structure.lattice.matrix)
        with patch(
            "phase_agent.science.structures.enumerate_layered_oxide_supercells._enumerate_icet_supercells",
            return_value=[FakeSupercell(cell)],
        ):
            result = enumerate_layered_oxide_supercells(
                "O3",
                self.structure,
                sizes=[4],
                p_small_list=[LAYERED_OXIDE_P_SMALL_RECOMMENDATIONS[0]],
                min_distance=3.1,
            )
        self.assertEqual(result, [])

    def test_custom_mixed_matrix_dimensions_can_be_selected_together(self):
        h = np.array([[1, 2, 0], [0, 6, 0], [0, 0, 1]])
        cell = h @ np.asarray(self.structure.lattice.matrix)
        with patch(
            "phase_agent.science.structures.enumerate_layered_oxide_supercells._enumerate_icet_supercells",
            return_value=[FakeSupercell(cell)],
        ):
            result = enumerate_layered_oxide_supercells(
                "o3",
                self.structure,
                sizes=[6],
                p_small_list=[
                    [[1, 0], [0, 2]],
                    LAYERED_OXIDE_P_SMALL_RECOMMENDATIONS[1],
                ],
                min_distance=1.0,
            )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["phase"], "O3")
        self.assertEqual(result[0]["H"], h.tolist())

    def test_empty_or_invalid_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "sizes"):
            enumerate_layered_oxide_supercells(
                "O3", self.structure, sizes=[], p_small_list=[[[1, 0], [0, 1]]]
            )


if __name__ == "__main__":
    unittest.main()

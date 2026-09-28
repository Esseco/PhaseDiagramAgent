"""Na-content phase rules and bounded layered H expansion."""

import unittest
from unittest.mock import patch

from config_layer.session.materialize_layered_h import materialize_layered_h
from scientific_layer.structures.boundary_utils import allowed_phases_at_x


class CompositionPhaseHConfigTests(unittest.TestCase):
    def test_phase_rules_are_composition_specific(self):
        rules = {"at_x": {"0": ["P3"], "1": ["O3"]},
                 "intermediate": ["O3", "P3", "OP2"]}
        self.assertEqual(allowed_phases_at_x(rules, 0), {"P3"})
        self.assertEqual(allowed_phases_at_x(rules, 1), {"O3"})
        self.assertEqual(allowed_phases_at_x(rules, "1/2"), {"O3", "P3", "OP2"})
        self.assertEqual(allowed_phases_at_x(["O3"], 0), {"O3"})

    def test_h_generation_is_bounded_and_saved_as_explicit_matrices(self):
        config = {"system": {"system_id": "layered_na_tm_oxide",
            "boundary": {"P": {"at_x": {"0": ["P3"], "1": ["O3"]},
                                "intermediate": ["O3"]}, "H": {},
                         "TM_ratio": {"Fe": 1, "Mn": 1}},
            "phase_references": {"O3": "O3.vasp", "P3": "P3.vasp"},
            "constraints": {},
            "H_generation": {"enabled": True, "size_min": 4, "size_max": 6,
                "size_step": 2, "selected_recommendation_indices": [0],
                "additional_containment_matrices": [], "min_distance_angstrom": 2.0}}}
        matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 1]]
        with patch("config_layer.session.materialize_layered_h.enumerate_layered_oxide_supercells",
                   return_value=[{"H": matrix}]) as enumerator:
            result = materialize_layered_h(config)
        self.assertEqual(result["system"]["boundary"]["H"], {"O3": [matrix], "P3": [matrix]})
        self.assertEqual(list(enumerator.call_args.args[2]), [4, 6])
        self.assertEqual(config["system"]["boundary"]["H"], {})

    def test_missing_containment_rule_is_not_guessed(self):
        config = {"system": {"system_id": "layered_na_tm_oxide", "boundary": {
            "P": ["O3"], "H": {}}, "phase_references": {"O3": "O3.vasp"},
            "H_generation": {"enabled": True, "size_min": 4, "size_max": 6,
                "size_step": 2}}}
        with self.assertRaisesRegex(ValueError, "至少一个"):
            materialize_layered_h(config)


if __name__ == "__main__":
    unittest.main()

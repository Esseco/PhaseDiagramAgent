import unittest

from scientific_layer.features.build_cost_record import build_cost_record
from execution_layer.budget.check_structure_cost_limits import check_structure_cost_limits


class SizeAwareCostTest(unittest.TestCase):
    def test_supercell_size_changes_proxy_cost_and_limits(self):
        config = {"reference_atoms": 10, "atom_exponent": 1.0, "scale": 1.0}
        small = build_cost_record(atom_count=20, evaluation_count=100, proxy_config=config)
        large = build_cost_record(atom_count=80, evaluation_count=100, proxy_config=config)
        self.assertEqual(large["proxy"]["value"], 4 * small["proxy"]["value"])
        check = check_structure_cost_limits(atom_count=80, det_H=16, proxy_cost=large["proxy"]["value"], limits={"max_atoms": 64, "max_det_H": 32, "max_proxy_cost_per_task": None})
        self.assertFalse(check["allowed"])
        self.assertIn("max_atoms", check["reasons"])


if __name__ == "__main__":
    unittest.main()

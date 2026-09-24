import unittest

from experiments.branch_surrogate.compare_branch_surrogates import compare_branch_surrogates
from experiments.branch_surrogate.default_surrogate_comparison_config import default_surrogate_comparison_config
from scientific_layer.surrogate_models.default_surrogate_config import default_surrogate_config


class SurrogateComparisonTest(unittest.TestCase):
    def test_unconfigured_methods_are_skipped_without_metrics(self):
        rows, feature_rows = [], []
        splits = ["train"] * 12 + ["validation"] * 4 + ["test"] * 4
        for i, split in enumerate(splits):
            rows.append({"branch_id": f"B{i}", "split": split, "split_group": f"G{i}", "branch": {}, "features": {"trial_energy_per_atom": i / 10}, "target": {"distance_to_fixed_hull": i / 20, "actual_cost": 1, "composition_group": "x"}, "important": i in {0, 15}})
            feature_rows.append({"branch_id": f"B{i}", "structure_records": [{"relax": {"status": "completed", "converged": True, "charged_cost": 1}, "manual": {"status": "completed", "values": {"f": float(i)}}, "embedding": {"status": "not_configured", "values": None}}]})
        dataset = {"task": "mc", "rows": rows, "split_manifest": {"fixed": True}}
        table = {"rows": feature_rows}
        comparison = default_surrogate_comparison_config(); comparison.update({"initial_revealed": 5, "selection_batch_size": 2})
        report = compare_branch_surrogates(dataset, table, surrogate_config=default_surrogate_config(), comparison_config=comparison)
        self.assertEqual(report["methods"]["manual_rf"]["status"], "completed")
        self.assertEqual(report["methods"]["embedding_rf"]["status"], "skipped")
        self.assertNotIn("mae", report["methods"]["embedding_rf"]["test"])
        self.assertEqual(report["methods"]["mattertune"]["status"], "skipped")
        self.assertFalse(report["production_policy_changed"])
        self.assertEqual(report["bohb_assessment"]["status"], "not_evaluated")


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest

from experiments.branch_surrogate.example_branch_dataset import run_example


class BranchSurrogateTest(unittest.TestCase):
    def test_example_preserves_negative_distance_and_split(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            result = run_example(directory)
        self.assertEqual(result["report"]["status"], "completed")
        self.assertEqual(result["report"]["ready_count"], 1)
        self.assertAlmostEqual(result["dataset"]["rows"][0]["target"]["distance_to_fixed_hull"], -0.1)
        self.assertIn(result["dataset"]["rows"][0]["split"], {"train", "validation", "test"})


if __name__ == "__main__":
    unittest.main()

import unittest

from phase_agent.science.surrogate_models.example_surrogate_models import run_example


class SurrogateModelsTest(unittest.TestCase):
    def test_mock_relax_and_unconfigured_optional_methods(self):
        result = run_example(root=".")
        record = result["feature_table"]["rows"][0]["structure_records"][0]
        self.assertEqual(record["manual"]["status"], "completed")
        self.assertEqual(record["embedding"]["status"], "not_configured")
        self.assertEqual(record["relax"]["branch_id"], "B1")
        self.assertTrue(record["relax"]["cache_hit"])
        self.assertEqual(record["relax"]["charged_cost"], 0.0)
        self.assertEqual(record["relax"]["charged_proxy_cost"], 0.0)
        self.assertEqual(result["first_run"]["feature_table"]["rows"][0]["structure_records"][0]["relax"]["charged_cost"], 2.0)
        self.assertEqual(result["first_run"]["feature_table"]["rows"][0]["structure_records"][0]["relax"]["charged_proxy_cost"], 10.0)
        self.assertEqual(result["comparison"]["methods"]["mattertune"]["status"], "not_implemented")


if __name__ == "__main__":
    unittest.main()

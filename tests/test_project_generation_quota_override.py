import unittest
from pathlib import Path

from phase_agent.configuration.session.project_config_json import expand_project_config, profile_digest


class ProjectGenerationQuotaOverrideTest(unittest.TestCase):
    def test_short_config_accepts_generation_selection_limit(self):
        document = {
            "profile": "layered-oxide-v1", "profile_digest": profile_digest(),
            "config": {"run": {"total_quota": 900, "batch_size": 300}},
            "overrides": {"run": {"generation_options": {
                "selection_config": {"max_per_framework": 8}
            }}},
        }
        config = expand_project_config(document, source=Path.cwd() / "unused.json")
        self.assertEqual(config["run"]["generation_options"]["selection_config"]
                         ["max_per_framework"], 8)

    def test_short_project_config_accepts_registered_strategy_quotas(self):
        document = {
            "profile": "layered-oxide-v1",
            "profile_digest": profile_digest(),
            "config": {"run": {"total_quota": 300, "batch_size": 96}},
            "overrides": {"round_strategy": {"rule_default": {
                "generation_quotas": {"coverage": 200, "composition": 40,
                                      "competing_phase": 20, "tm_ordering": 20,
                                      "periodic_extension": 20},
            }}},
        }
        config = expand_project_config(document, source=Path.cwd() / "unused.json")
        self.assertEqual(config["run"]["batch_size"], 96)
        self.assertEqual(sum(config["round_strategy"]["rule_default"]
                             ["generation_quotas"].values()), 300)


if __name__ == "__main__":
    unittest.main()

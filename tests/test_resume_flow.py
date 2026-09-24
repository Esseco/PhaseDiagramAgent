from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data_layer.ledger.phase_data_manager import PhaseDataManager
from run.default_run_config import default_run_config
from run.resume_pipeline import resume_pipeline
from run.run_pipeline import run_pipeline


class ResumeFlowTest(unittest.TestCase):
    def test_budget_stop_resume_and_no_duplicate_submission(self):
        H = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        manager = PhaseDataManager({"P": ["O3"], "H": {"O3": [H]}, "TM_ratio": {"Fe": 1}})
        branch_id = manager.add_branch(P="O3", H=H, x=1, T=["Fe"])
        structure_id = manager.add_structure(branch_id=branch_id, arrangement={"V": [1]}, source_path="S-1.vasp")
        generated = {"registered": [{"structure_id": structure_id}]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = default_run_config()
            config.update({"structure_directory": str(root / "structures"), "state_path": str(root / "state.json"), "ledger_path": str(root / "ledger.json"), "phase_diagram_directory": str(root / "phase"), "total_quota": 1, "batch_size": 1, "initial_states_per_branch": 1})
            config["budgets"]["total_relative_cost"] = 0.0
            with patch("run.search_iteration.run_branch_generation", return_value=generated):
                stopped = run_pipeline(manager, {}, config)
            self.assertEqual(stopped["state"]["run_status"], "budget_exhausted")
            self.assertEqual(stopped["dispatched"], [])

            empty_generation = {"registered": []}
            dispatcher = lambda task: {"status": "pending"}
            with patch("run.search_iteration.run_branch_generation", return_value=empty_generation):
                resumed = resume_pipeline(config, {}, dispatcher=dispatcher, additional_budget=0.05)
            self.assertEqual(len(resumed["dispatched"]), 1)
            self.assertEqual(len(resumed["state"]["pending_tasks"]), 1)
            self.assertEqual(resumed["state"]["budget_limits"]["total_relative_cost"], 0.05)

            with patch("run.search_iteration.run_branch_generation", return_value=empty_generation):
                repeated = resume_pipeline(resumed["effective_config"], {}, dispatcher=dispatcher)
            self.assertEqual(repeated["dispatched"], [])
            self.assertEqual(len(repeated["state"]["pending_tasks"]), 1)


if __name__ == "__main__":
    unittest.main()

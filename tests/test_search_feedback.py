"""一轮搜索反馈的轻量模拟，不启动正式计算。"""

from __future__ import annotations

import unittest
import tempfile
import csv
from pathlib import Path
from unittest.mock import patch

from phase_agent.persistence.ledger.phase_data_manager import PhaseDataManager
from phase_agent.analysis.feedback.calculate_search_reward import calculate_search_reward
from phase_agent.analysis.convergence.check_global_convergence import check_global_convergence
from phase_agent.analysis.phase.update_phase_diagram import update_phase_diagram


class SearchFeedbackTest(unittest.TestCase):
    def test_versioned_phase_csv_keeps_unknown_na_check(self):
        records = [
            {"record_id": "na", "energy_method": "mlip", "composition": {"Na": 1},
             "energy": 0.0, "energy_unit": "eV", "source_version": "m1"},
            {"record_id": "o", "energy_method": "mlip", "composition": {"O": 1},
             "energy": 0.0, "energy_unit": "eV", "source_version": "m1"},
            {"record_id": "oxide", "energy_method": "mlip",
             "composition": {"Na": 2, "O": 1}, "energy": -3.0,
             "energy_unit": "eV", "source_version": "m1"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            result = update_phase_diagram(records, output_directory=directory)
            snapshot = result["diagrams"]["mlip"]
            csv_path = Path(snapshot["csv_path"])
            self.assertTrue(csv_path.is_file())
            self.assertEqual(csv_path.name, "phase_diagram.csv")
            archive = Path(snapshot["archive_csv_path"])
            self.assertIn(snapshot["version"], archive.name)
            self.assertEqual(csv_path.read_bytes(), archive.read_bytes())
            with csv_path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            oxide = next(row for row in rows if row["record_id"] == "oxide")
            self.assertEqual(oxide["na_layer_status"], "missing_final_structure")
            self.assertEqual(oxide["x_Na_per_O2"], "4.0000000000")

    def test_phase_diagrams_are_separate_and_reward_is_idempotent(self):
        records = []
        for method in ("mlip", "dft"):
            records.extend(
                [
                    {
                        "record_id": f"{method}-na",
                        "energy_method": method,
                        "composition": {"Na": 1},
                        "energy": 0.0,
                        "energy_unit": "eV",
                    },
                    {
                        "record_id": f"{method}-o",
                        "energy_method": method,
                        "composition": {"O": 1},
                        "energy": 0.0,
                        "energy_unit": "eV",
                    },
                    {
                        "record_id": f"{method}-oxide",
                        "energy_method": method,
                        "composition": {"Na": 2, "O": 1},
                        "energy": -3.0,
                        "energy_unit": "eV",
                    },
                ]
            )
        diagrams = update_phase_diagram(records)
        self.assertEqual(diagrams["diagrams"]["mlip"]["status"], "completed")
        self.assertEqual(diagrams["diagrams"]["dft"]["status"], "completed")
        before = diagrams["diagrams"]["dft"]
        after = {**before, "entries": [dict(item) for item in before["entries"]]}
        reward = calculate_search_reward(
            before, after, batch_id="B1", task_ids=["T1"], actual_cost=2.0
        )
        duplicate = calculate_search_reward(
            before,
            after,
            batch_id="B1",
            task_ids=["T1"],
            actual_cost=2.0,
            reward_state=reward["state"],
        )
        self.assertEqual(duplicate["status"], "already_accounted")

    def test_budget_exhaustion_is_not_convergence(self):
        result = check_global_convergence(
            {
                "tasks": [],
                "hull_changes": [1.0],
                "budget_remaining": 0,
                "dft_validation_fraction": 0,
                "coverage_fraction": 0,
                "open_branches": 2,
            }, rules={"hull_change_tolerance": .003, "stable_model_update_epochs": 2,
                      "final_energy_mae_tolerance": .003, "minimum_recent_dft_checks": 1}
        )
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertFalse(result["converged"])




if __name__ == "__main__":
    unittest.main()

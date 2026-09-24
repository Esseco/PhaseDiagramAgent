"""一轮搜索反馈的轻量模拟，不启动正式计算。"""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from data_layer.ledger.phase_data_manager import PhaseDataManager
from analysis_layer.feedback.calculate_search_reward import calculate_search_reward
from analysis_layer.convergence.check_global_convergence import check_global_convergence
from run.search_iteration import run_search_iteration
from analysis_layer.phase.update_phase_diagram import update_phase_diagram


class SearchFeedbackTest(unittest.TestCase):
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

    def test_one_iteration_leaves_unfinished_task_pending(self):
        H = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        manager = PhaseDataManager(
            {"P": ["O3"], "H": {"O3": [H]}, "TM_ratio": {"Fe": 1}}
        )
        branch_id = manager.add_branch(P="O3", H=H, x=1, T=["Fe"])
        structure_id = manager.add_structure(
            branch_id=branch_id,
            arrangement={"V": [1]},
            source_path="S-1.vasp",
        )
        generated = {
            "registered": [{"structure_id": structure_id, "structure_path": "S-1.vasp"}]
        }
        with patch(
            "run.search_iteration.run_branch_generation",
            return_value=generated,
        ):
            result = run_search_iteration(
                manager,
                {},
                None,
                structure_directory="unused",
                total_quota=5,
                batch_size=1,
                initial_states_per_branch=1,
                seed=9,
            )
        self.assertEqual(result["state"]["iteration"], 1)
        self.assertEqual(result["state"]["pending_tasks"][0]["status"], "pending")
        self.assertEqual(result["decision"]["seed"], 9)

        with patch(
            "run.search_iteration.run_branch_generation",
            return_value=generated,
        ):
            repeated = run_search_iteration(
                manager,
                {},
                result["state"],
                structure_directory="unused",
                total_quota=5,
                batch_size=1,
                initial_states_per_branch=1,
                seed=10,
            )
        self.assertEqual(len(repeated["state"]["pending_tasks"]), 1)
        self.assertEqual(repeated["dispatched"], [])

    def test_iteration_state_can_be_saved_and_restored(self):
        H = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        manager = PhaseDataManager(
            {"P": ["O3"], "H": {"O3": [H]}, "TM_ratio": {"Fe": 1}}
        )
        generated = {"registered": []}
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            state_path = Path(directory) / "state.json"
            with patch(
                "run.search_iteration.run_branch_generation",
                return_value=generated,
            ):
                run_search_iteration(
                    manager,
                    {},
                    None,
                    structure_directory="unused",
                    total_quota=5,
                    batch_size=1,
                    initial_states_per_branch=1,
                    seed=1,
                    state_path=state_path,
                )
                restored = run_search_iteration(
                    manager,
                    {},
                    state_path,
                    structure_directory="unused",
                    total_quota=5,
                    batch_size=1,
                    initial_states_per_branch=1,
                    seed=2,
                )
            self.assertEqual(restored["state"]["iteration"], 2)


if __name__ == "__main__":
    unittest.main()

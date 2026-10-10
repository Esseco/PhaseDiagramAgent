"""无耗时计算的模拟接口验证。"""

from __future__ import annotations

import unittest

from pymatgen.core import Lattice, Structure

from phase_agent.persistence.ledger.collect_calculation_results import collect_calculation_results
from phase_agent.decisions.calculation.decide_next_calculation import decide_next_calculation
from phase_agent.science.dft.run_singlepoint import run_dft_singlepoint
from phase_agent.science.mlip.run_relax import run_mlip_relax
from phase_agent.persistence.ledger.phase_data_manager import PhaseDataManager


class CalculationWorkflowTest(unittest.TestCase):
    def setUp(self):
        H = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.manager = PhaseDataManager(
            {"P": ["O3"], "H": {"O3": [H]}, "TM_ratio": {"Fe": 1}}
        )
        branch = self.manager.add_branch(
            P="O3", H=H, x=1, T=["Fe"], composition={"Fe": 1}
        )
        self.structure_id = self.manager.add_structure(
            branch_id=branch, arrangement={"V": [1]}
        )
        self.structure = Structure(Lattice.cubic(3), ["Fe"], [[0, 0, 0]])

    def test_missing_backend_is_not_completed(self):
        result = run_dft_singlepoint(self.structure)
        saved = collect_calculation_results(self.manager, self.structure_id, result)
        self.assertEqual(saved["status"], "not_configured")
        self.assertIsNone(saved["result_id"])
        self.assertFalse(
            self.manager.data["structures"][self.structure_id]["stage_history"][
                "dft_single_point"
            ]
        )

    def test_mock_backend_and_decision(self):
        def backend(**kwargs):
            return {
                "structure": kwargs["structure"],
                "energy": -1.2,
                "energy_unit": "eV",
                "converged": True,
                "cost": {"steps": 3},
            }

        result = run_mlip_relax(
            self.structure, backend=backend, model_path="mock.model"
        )
        saved = collect_calculation_results(self.manager, self.structure_id, result)
        self.assertIsNotNone(saved["result_id"])
        self.assertEqual(saved["phase_record"]["energy_method"], "mlip")
        decision = decide_next_calculation(
            self.manager.data["structures"][self.structure_id],
            rules={"skip_stages": ["simple_check"]},
        )
        self.assertEqual(decision["stage"], "deep_search")

    def test_pending_and_repeat_rules(self):
        record = self.manager.data["structures"][self.structure_id]
        record["metadata"] = {
            "calculation_attempts": [
                {"task_id": "T-1", "stage": "simple_check", "status": "pending"}
            ]
        }
        self.assertEqual(decide_next_calculation(record)["action"], "run")
        record["metadata"]["calculation_attempts"] = []
        self.manager.record_result(
            structure_id=self.structure_id, stage="simple_check", converged=True
        )
        decision = decide_next_calculation(
            record, rules={"repeat_stages": ["simple_check"]}
        )
        self.assertEqual(decision["reason"], "repeat_rule")

    def test_task_status_update_allows_next_stage(self):
        pending = {
            "task_id": "T-2",
            "stage": "simple_check",
            "status": "pending",
            "converged": None,
        }
        collect_calculation_results(self.manager, self.structure_id, pending)
        completed = {
            **pending,
            "status": "completed",
            "converged": True,
            "outputs": {},
        }
        collect_calculation_results(self.manager, self.structure_id, completed)
        record = self.manager.data["structures"][self.structure_id]
        attempts = record["metadata"]["calculation_attempts"]
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]["status"], "completed")
        self.assertEqual(decide_next_calculation(record)["stage"], "relax_and_feature")


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from analysis_layer.convergence.build_convergence_evidence import build_convergence_evidence
from scientific_layer.qbc.create_qbc_evaluator import create_qbc_evaluator
from execution_layer.workflows.create_model_update_handler import create_model_update_handler


class _Manager:
    def __init__(self):
        self.data = {
            "branches": {"B1": {}, "B2": {}},
            "structures": {
                "S1": {"branch_id": "B1", "stage_history": {"deep_search": [{}], "dft_single_point": [{}]}},
                "S2": {"branch_id": "B2", "stage_history": {"deep_search": [], "dft_single_point": []}},
            },
        }


class ActiveLearningLifecycleTest(unittest.TestCase):
    def test_qbc_adapter_only_returns_uncertainty(self):
        structure = [object(), object()]
        committee = {"loaded_members": [
            {"model_id": "m1", "model": 1.0},
            {"model_id": "m2", "model": 2.0},
        ]}
        evaluator = create_qbc_evaluator(
            committee,
            predictor=lambda structure, model, member: {
                "energy": model,
                "forces": np.full((len(structure), 3), model),
            },
        )
        result = evaluator({"structure": structure})
        self.assertEqual(result["status"], "completed")
        self.assertIn("energy_std", result)
        self.assertNotIn("action", result)

    def test_model_lifecycle_requires_validation_before_activation(self):
        manager = _Manager()
        state = {
            "iteration": 3,
            "active_model_version": "m1",
            "active_model": {"version": "m1"},
            "new_dft_records": [{"structure_id": "S1", "status": "completed", "converged": True, "checks_passed": True, "energy": -1.0}],
            "phase_diagrams": {"mlip": {"record_id": "H1", "model_version": "m1", "validity": "valid"}},
        }
        handler = create_model_update_handler(
            trainer=lambda **kwargs: {"artifact": "candidate"},
            validation_evaluator=lambda model, data: {
                "energy_mae": 0.1 if model.get("version") == "m1" else 0.05,
                "force_rmse": 0.2 if model.get("version") == "m1" else 0.1,
                "critical_failure_fraction": 0.0, "near_hull_ranking_reversals": 0,
            },
            validation_data_provider=lambda **kwargs: [{"id": "V1"}],
            candidate_provider=lambda **kwargs: [],
        )
        result = handler(
            trigger={"action": "RETRAIN_MLIP"}, state=state, manager=manager,
            config={"mlip": {"version": "m1"}, "mlip_finetune": {}},
        )
        self.assertEqual(result["status"], "needs_user_confirmation")
        self.assertEqual(result["state"]["active_model_version"], "m1")
        self.assertEqual(result["state"]["new_dft_records"], state["new_dft_records"])
        self.assertFalse(result["validation"]["passed"])
        confirmed_config = {"mlip": {"version": "m1"}, "mlip_finetune": {
            "validation": {"max_energy_mae": .2, "max_critical_failure_fraction": .1,
                           "max_near_hull_ranking_reversals": 1},
            "activation": {"requires_separate_approval": True}}}
        result = handler(trigger={"action": "RETRAIN_MLIP"}, state=state,
                         manager=manager, config=confirmed_config)
        self.assertEqual(result["status"], "awaiting_activation_approval")
        self.assertTrue(result["validation"]["passed"])
        pending_state = result["state"]
        refused = handler(trigger={"action": "ACTIVATE_CANDIDATE_MODEL",
            "candidate_model_version": "mlip-candidate-000003"}, state=pending_state,
            manager=manager, config={"mlip": {"version": "m1"}})
        self.assertEqual(refused["status"], "awaiting_user_approval")
        self.assertEqual(refused["state"]["active_model_version"], "m1")
        result = handler(trigger={"action": "ACTIVATE_CANDIDATE_MODEL",
            "candidate_model_version": "mlip-candidate-000003",
            "user_approval_reason": "独立验证改善，确认启用"}, state=pending_state,
            manager=manager, config={"mlip": {"version": "m1"}})
        self.assertEqual(result["status"], "activated")
        self.assertEqual(result["state"]["active_model_version"], "mlip-candidate-000003")
        self.assertEqual(result["state"]["phase_diagrams"]["mlip"]["validity"], "stale")
        self.assertEqual(result["state"]["new_dft_records"], [])

    def test_convergence_evidence_uses_ledger_and_pending_tasks(self):
        evidence = build_convergence_evidence(
            {"pending_tasks": [{"status": "running"}], "hull_changes": [0.1]},
            _Manager(), budget_remaining=7,
        )
        self.assertEqual(evidence["dft_validation_fraction"], 0.5)
        self.assertEqual(evidence["dft_validation_count"], 1)
        self.assertEqual(evidence["coverage_fraction"], 1.0)
        self.assertEqual(evidence["open_branches"], 1)
        self.assertEqual(evidence["budget_remaining"], 7)


if __name__ == "__main__":
    unittest.main()

import unittest

from phase_agent.analysis.convergence.check_global_convergence import check_global_convergence
from phase_agent.configuration.defaults.default_layered_search_config import default_layered_search_config
from phase_agent.configuration.session.apply_config_revision import apply_config_revision
from phase_agent.configuration.session.confirm_config_snapshot import confirm_config_snapshot
from phase_agent.configuration.session.create_config_draft import create_config_draft
from phase_agent.tools.state.reconcile_task_results import reconcile_task_results
from phase_agent.science.training.validate_mlip import validate_mlip


RULES = {"hull_change_tolerance": .003, "stable_model_update_epochs": 2,
         "final_energy_mae_tolerance": .01, "minimum_recent_dft_checks": 1}


class MCCostConvergenceModelPolicyTest(unittest.TestCase):
    def _reserved(self, task_id="T1", task_key="K1"):
        return {"tasks": [{"task_id": task_id, "task_key": task_key, "status": "pending"}],
                "budget_reservations": {task_key: {"status": "reserved", "reserved_cost": 10,
                                                    "stage": "deep_search"}},
                "reserved_relative_cost": 10}

    def test_patience_early_stop_keeps_parameters_and_unknown_actual_cost(self):
        result = reconcile_task_results(self._reserved(), [{
            "task_id": "T1", "task_key": "K1", "status": "completed",
            "max_mc_steps": 100, "requested_max_mc_steps": 100,
            "patience_steps": 12, "min_improvement": .001,
            "actual_mc_steps": 20, "stop_reason": "patience",
        }])
        task = result["state"]["tasks"][0]
        reservation = result["state"]["budget_reservations"]["K1"]
        self.assertEqual(task["patience_steps"], 12)
        self.assertEqual(task["actual_mc_steps"], 20)
        self.assertIsNone(task["actual_cost"])
        self.assertEqual(task["estimated_cost"], 2)
        self.assertEqual(reservation["released_cost"], 8)
        self.assertFalse(reservation["actual_cost_known"])

    def test_scheduler_gpu_hours_failure_cost_and_duplicate_settlement(self):
        incoming = {"task_id": "T1", "task_key": "K1", "status": "failed",
                    "actual_cost": 3, "actual_cost_basis": "scheduler_billing_record",
                    "actual_mc_steps": 7, "stop_reason": "backend_error",
                    "job_accounting": {"source": "sacct", "gpu_count": 2,
                                       "elapsed_seconds": 1800, "job_id": "99"}}
        first = reconcile_task_results(self._reserved(), [incoming])
        second = reconcile_task_results(first["state"], [incoming])
        reservation = first["state"]["budget_reservations"]["K1"]
        self.assertEqual(reservation["actual_cost"], 3)
        self.assertEqual(reservation["actual_gpu_core_hours"], 1)
        self.assertEqual(reservation["released_cost"], 7)
        self.assertEqual(second["state"]["budget_usage"]["total_relative_cost"], 3)

    def test_same_model_mc_rounds_do_not_count_as_epochs_and_missing_is_not_zero(self):
        state = {"model_update_epochs": [
                    {"model_version": "m1", "hull_change": .001, "ground_state_unchanged": True},
                    {"model_version": "m1", "hull_change": .001, "ground_state_unchanged": True}],
                 "final_frame_dft_errors": [{"error_ev_per_atom": .001}], "budget_remaining": 10}
        result = check_global_convergence(state, rules=RULES)
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["model_update_epochs_counted"], ["m1"])
        state["model_update_epochs"].append(
            {"model_version": "m2", "hull_change": None, "ground_state_unchanged": True})
        self.assertEqual(check_global_convergence(state, rules=RULES)["status"], "insufficient_evidence")

    def test_low_coverage_numeric_success_waits_for_user(self):
        state = {"model_update_epochs": [
                    {"model_version": "m1", "hull_change": .001, "ground_state_unchanged": True},
                    {"model_version": "m2", "hull_change": .002, "ground_state_unchanged": True}],
                 "final_frame_dft_errors": [{"error_ev_per_atom": .005}],
                 "coverage": {"fraction": .2}, "budget_remaining": 10}
        result = check_global_convergence(state, rules=RULES)
        self.assertEqual(result["status"], "numerical_criteria_satisfied")
        self.assertEqual(result["coverage_risk"]["status"], "incomplete")
        self.assertFalse(result["converged"])

    def test_convergence_rules_are_versioned_and_user_change_creates_new_version(self):
        config = default_layered_search_config()
        first = create_config_draft(config)
        first = apply_config_revision(first, {"calculation.mlip_version": "m1",
                                              "dft.parameters": {"encut": 520}})
        first = confirm_config_snapshot(first, user_confirmed=True)
        second = create_config_draft(first["confirmed_snapshot"]["config"])
        second = apply_config_revision(second, {"convergence.stable_model_update_epochs": 3})
        second = confirm_config_snapshot(second, user_confirmed=True)
        self.assertNotEqual(first["confirmed_snapshot"]["config_version"],
                            second["confirmed_snapshot"]["config_version"])
        self.assertEqual(second["confirmed_snapshot"]["config"]["convergence"]["stable_model_update_epochs"], 3)

    def test_small_mae_rise_allowed_severe_anomaly_blocked_and_missing_limits_stop(self):
        def evaluator(model, data):
            return {"energy_mae": .10 if model["version"] == "old" else .11,
                    "force_rmse": .2, "critical_failure_fraction": 0,
                    "near_hull_ranking_reversals": 0}
        limits = {"max_energy_mae": .2, "max_critical_failure_fraction": .2,
                  "max_near_hull_ranking_reversals": 0}
        accepted = validate_mlip({"version": "old"}, {"version": "new", "dataset_version": "D2"},
                                 [{}], evaluator=evaluator, criteria=limits,
                                 validation_data_version="V2")
        self.assertTrue(accepted["passed"])
        self.assertEqual(accepted["benefit_status"], "no_distinguishable_benefit")
        def severe(model, data):
            value = evaluator(model, data)
            if model["version"] == "new": value["critical_failure_fraction"] = .8
            return value
        blocked = validate_mlip({"version": "old"}, {"version": "new"}, [{}],
                                evaluator=severe, criteria=limits, validation_data_version="V2")
        self.assertFalse(blocked["passed"])
        self.assertIn("critical_failures", blocked["anomalies"])
        missing = validate_mlip({"version": "old"}, {"version": "new"}, [{}],
                                evaluator=evaluator, criteria={}, validation_data_version="V2")
        self.assertEqual(missing["status"], "needs_user_confirmation")


if __name__ == "__main__":
    unittest.main()

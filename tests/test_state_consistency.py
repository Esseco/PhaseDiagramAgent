import tempfile
import unittest

from data_layer.models.activate_validated_model import activate_validated_model
from analysis_layer.state.summarize_agent_state import summarize_agent_state
from config_layer.defaults.default_budget_rules import default_budget_rules
from execution_layer.budget.expire_budget_reservations import expire_budget_reservations
from execution_layer.budget.reserve_budget import reserve_budget
from execution_layer.budget.settle_budget import settle_budget
from scientific_layer.structures.validate_structure_transition import validate_structure_transition
from analysis_layer.convergence.check_global_convergence import check_global_convergence
from config_layer.runtime.authorize_budget_extension import authorize_budget_extension


class StateConsistencyTest(unittest.TestCase):
    def test_new_convergence_uses_final_stable_frame_error(self):
        state = {"tasks": [], "model_update_epochs": [
            {"model_version": "m1", "hull_change": .002, "ground_state_unchanged": True},
            {"model_version": "m2", "hull_change": .001, "ground_state_unchanged": True},
        ], "final_frame_dft_errors": [{"error_ev_per_atom": .003}]}
        rules = {"hull_change_tolerance": .003, "stable_model_update_epochs": 2,
                 "final_energy_mae_tolerance": .003, "minimum_recent_dft_checks": 1}
        numerical = check_global_convergence(state, rules=rules)
        self.assertTrue(numerical["criteria_satisfied"])
        self.assertEqual(numerical["status"], "numerical_criteria_satisfied")
        state["user_accepted_convergence"] = True
        self.assertTrue(check_global_convergence(state, rules=rules)["converged"])
        state.pop("user_accepted_convergence")
        state["final_frame_dft_errors"][-1]["error_ev_per_atom"] = .0031
        self.assertFalse(check_global_convergence(state, rules=rules)["converged"])

    def test_confirmed_budget_extension_rebinds_without_resetting_state(self):
        old = {"budgets": {"total_relative_cost": 10}, "system": {"name": "x"}}
        state = {"confirmed_config_version": "v1", "confirmed_config": old, "tasks": [{"task_id": "done"}]}
        result = authorize_budget_extension(state, {"config_version": "v2", "config": {"budgets": {"total_relative_cost": 20}, "system": {"name": "x"}}}, user_approved=True)
        self.assertEqual(result["status"], "extended")
        self.assertEqual(result["state"]["tasks"], state["tasks"])
        self.assertEqual(result["state"]["confirmed_config_version"], "v2")

    def test_concurrent_reservation_and_duplicate_settlement(self):
        limits = default_budget_rules(); limits["total_relative_cost"] = 10; limits["stage_limits"]["deep_search"]["max_cost"] = 10
        first = reserve_budget({}, task_key="A", stage="deep_search", amount=7, limits=limits)
        second = reserve_budget(first["state"], task_key="B", stage="deep_search", amount=5, limits=limits)
        self.assertEqual(first["status"], "reserved"); self.assertEqual(second["status"], "rejected")
        settled = settle_budget(first["state"], task_key="A", settlement_id="R1", task_status="failed", actual_cost=3, failure_reason="backend")
        repeated = settle_budget(settled["state"], task_key="A", settlement_id="R1", task_status="failed", actual_cost=3)
        self.assertEqual(settled["reservation"]["released_cost"], 4); self.assertEqual(repeated["status"], "already_settled")
        self.assertEqual(repeated["state"]["budget_usage"]["total_relative_cost"], 3)

    def test_timeout_model_switch_structure_change_and_convergence_unknown(self):
        limits = default_budget_rules()
        reserved = reserve_budget({}, task_key="T", stage="deep_search", amount=4, limits=limits, timeout_at="2020-01-01T00:00:00+00:00")["state"]
        reserved["budget_reservations"]["T"]["status"] = "submitted"
        expired = expire_budget_reservations(reserved, now="2021-01-01T00:00:00+00:00")["state"]
        self.assertEqual(expired["budget_usage"]["total_relative_cost"], 4)
        state = {"active_model_version": "m1", "energy_records": [{"record_id": "E1", "model_version": "m1", "validity": "valid"}], "feature_records": [{"record_id": "F1", "model_version": "m1", "validity": "valid"}], "surrogate_predictions": [], "phase_diagrams": [{"record_id": "H1", "model_version": "m1", "validity": "valid"}]}
        switched = activate_validated_model(state, {"version": "m2"}, {"status": "completed", "passed": True, "validation_data_version": "v1"}, approval_reason="user accepted mock")
        self.assertEqual(switched["state"]["energy_records"][0]["validity"], "stale")
        summary = summarize_agent_state(switched["state"]); self.assertEqual(len(summary["stale_or_refresh_required"]), 3)
        transition = validate_structure_transition({"P": "O3", "H": [[1]], "x": .5, "T": ["Fe", "Mn"]}, {"P": "P3", "H": [[2]], "x": .5, "T": ["Fe", "Mn"]})
        self.assertTrue(transition["valid"]); self.assertTrue(transition["structure_family_changed"])
        invalid = validate_structure_transition({"P": "O3", "x": .5, "T": ["Fe", "Mn"]}, {"P": "P3", "x": .75, "T": ["Fe", "Mn"]})
        self.assertFalse(invalid["valid"])
        convergence = check_global_convergence({"budget_remaining": 0}, rules={
            "hull_change_tolerance": .003, "stable_model_update_epochs": 2,
            "final_energy_mae_tolerance": .003, "minimum_recent_dft_checks": 1})
        self.assertEqual(convergence["status"], "budget_exhausted"); self.assertFalse(convergence["converged"]); self.assertTrue(convergence["missing_evidence"])


if __name__ == "__main__": unittest.main()

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
        old = {"budgets": {"total_relative_cost": 10}, "system": {"name": "x"},
               "round_strategy": {"maximum_mc_budget": 1000}}
        state = {"confirmed_config_version": "v1", "confirmed_config": old, "tasks": [{"task_id": "done"}]}
        result = authorize_budget_extension(state, {"config_version": "v2", "config": {
            "budgets": {"total_relative_cost": 20}, "system": {"name": "x"},
            "round_strategy": {"maximum_mc_budget": 5000},
        }}, user_approved=True)
        self.assertEqual(result["status"], "extended")
        self.assertEqual(result["state"]["tasks"], state["tasks"])
        self.assertEqual(result["state"]["confirmed_config_version"], "v2")
        self.assertEqual(result["budget_changes"], [
            {"field": "budgets.total_relative_cost", "from": 10, "to": 20},
            {"field": "round_strategy.maximum_mc_budget", "from": 1000, "to": 5000},
        ])

    def test_approved_budget_migration_is_a_separate_non_executing_step(self):
        from types import SimpleNamespace
        from run.main import run_workflow

        old = {"system": {"id": "test"}, "budgets": {"total_relative_cost": 10},
               "round_strategy": {"maximum_mc_budget": 1000}, "agent": {"allowed_tools": []}}
        new = {"system": {"id": "test"}, "budgets": {"total_relative_cost": 20},
               "round_strategy": {"maximum_mc_budget": 5000}, "agent": {"allowed_tools": []}}
        state = {"confirmed_config_version": "v1", "confirmed_config": old,
                 "tasks": [{"task_id": "old-task", "status": "completed", "config_version": "v1"}],
                 "budget_usage": {"total_relative_cost": 2}}
        result = run_workflow(
            SimpleNamespace(data={"branches": {}, "structures": {}}), {}, {},
            {"status": "confirmed", "confirmed_snapshot": {"config_version": "v2", "config": new}},
            state=state, approve_budget_extension=True,
        )
        self.assertEqual(result["status"], "config_migrated")
        self.assertFalse(result["submitted"])
        self.assertEqual(result["state"]["confirmed_config_version"], "v2")
        self.assertEqual(result["state"]["tasks"][0]["config_version"], "v1")
        self.assertEqual(result["state"]["budget_remaining"], 18)

    def test_config_migration_rejects_scientific_mlip_setting_changes_with_paths(self):
        old = {"budgets": {"total_relative_cost": 10},
               "mlip": {"name": "mace-mh-1", "mace_head": None, "relax_parameters": None}}
        new = {"budgets": {"total_relative_cost": 20},
               "mlip": {"name": "mace-mh-1", "mace_head": "omat_pbe",
                        "relax_parameters": {"fmax": 0.05}}}
        state = {"confirmed_config_version": "v1", "confirmed_config": old,
                 "tasks": [{"task_id": "done", "config_version": "v1"}]}
        result = authorize_budget_extension(
            state, {"config_version": "v2", "config": new}, user_approved=True,
        )
        self.assertEqual(result["status"], "rejected_non_budget_change")
        self.assertEqual(result["changed_fields"], ["mlip.mace_head", "mlip.relax_parameters"])
        self.assertEqual(result["state"]["confirmed_config_version"], "v1")
        self.assertEqual(result["state"]["tasks"], state["tasks"])

    def test_approved_migration_accepts_only_evidenced_legacy_mace_defaults(self):
        old = {
            "budgets": {"total_relative_cost": 20000},
            "mlip": {"name": "mace-mh-1", "mace_head": None, "relax_parameters": None},
        }
        new = {
            "budgets": {"total_relative_cost": 20000},
            "mlip": {
                "name": "mace-mh-1", "mace_head": "omat_pbe",
                "relax_parameters": {
                    "fmax": .05, "mace_default_dtype": "float64",
                    "relax_cell": True, "relax_steps": 150,
                },
            },
        }
        task = {
            "task_id": "RELAX-1", "stage": "relax_and_feature",
            "status": "completed", "config_version": "v1",
            "model_version": "mace-mh-1", "parameters": {},
            "outputs": {
                "status": "completed", "mace_head": "omat_pbe",
                "fmax_target_ev_per_angstrom": .05, "cell_relaxed": True,
                "relax_steps_used": 150,
            },
        }
        state = {"confirmed_config_version": "v1", "confirmed_config": old,
                 "tasks": [task]}

        result = authorize_budget_extension(
            state, {"config_version": "v2", "config": new}, user_approved=True,
        )

        self.assertEqual(result["status"], "extended")
        self.assertEqual(result["state"]["tasks"][0]["config_version"], "v1")
        self.assertEqual(result["compatibility"]["relax_task_count"], 1)
        self.assertIn("mlip.relax_parameters.mace_default_dtype",
                      result["compatibility"]["unrecorded_fields"])
        self.assertEqual(
            result["state"]["config_migrations"][-1]["type"],
            "approved_migration_with_verified_legacy_mace_defaults",
        )

    def test_legacy_mace_migration_rejects_unverified_or_changed_relax_results(self):
        old = {"budgets": {"total_relative_cost": 20},
               "mlip": {"name": "mace-mh-1", "mace_head": None, "relax_parameters": None}}
        new = {"budgets": {"total_relative_cost": 20},
               "mlip": {"name": "mace-mh-1", "mace_head": "omat_pbe",
                        "relax_parameters": {
                            "fmax": .05, "mace_default_dtype": "float64",
                            "relax_cell": True, "relax_steps": 150,
                        }}}
        task = {
            "task_id": "RELAX-1", "stage": "relax_and_feature", "status": "completed",
            "model_version": "mace-mh-1", "outputs": {
                "status": "completed", "mace_head": "omat_pbe",
                "fmax_target_ev_per_angstrom": .05, "cell_relaxed": True,
                "relax_steps_used": 40,
            },
        }
        task["outputs"]["mace_head"] = "different-head"
        state = {"confirmed_config_version": "v1", "confirmed_config": old,
                 "tasks": [task]}
        result = authorize_budget_extension(
            state, {"config_version": "v2", "config": new}, user_approved=True,
        )
        self.assertEqual(result["status"], "rejected_non_budget_change")
        self.assertEqual(result["state"]["confirmed_config_version"], "v1")
        self.assertTrue(result["compatibility_rejection"])

    def test_budget_migration_does_not_change_structure_search_bounds(self):
        old = {"budgets": {
            "total_relative_cost": 10,
            "structure_limits": {"max_det_H": 24},
        }}
        new = {"budgets": {
            "total_relative_cost": 20,
            "structure_limits": {"max_det_H": 64},
        }}
        state = {"confirmed_config_version": "v1", "confirmed_config": old,
                 "tasks": [{"task_id": "done", "status": "completed"}]}
        result = authorize_budget_extension(
            state, {"config_version": "v2", "config": new}, user_approved=True,
        )
        self.assertEqual(result["status"], "rejected_non_budget_change")
        self.assertIn("budgets.structure_limits.max_det_H", result["changed_fields"])

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

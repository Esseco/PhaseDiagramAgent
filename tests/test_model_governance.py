import json
import tempfile
import unittest
from pathlib import Path

from phase_agent.analysis.feedback.assess_model_refresh import assess_model_refresh
from phase_agent.persistence.models.rollback_model import rollback_model
from phase_agent.tools.workflows.create_model_update_handler import create_model_update_handler
from phase_agent.runtime.step_runner import _runtime_config
from phase_agent.science.training.assess_initial_mlip import assess_initial_mlip
from phase_agent.science.training.validate_mlip import validate_mlip
from phase_agent.tools.state.restart_failed_task import restart_failed_task
from phase_agent.configuration.defaults.default_budget_rules import default_budget_rules


class ModelGovernanceTest(unittest.TestCase):
    def test_failed_retry_is_new_idempotent_budgeted_task(self):
        failed = {"task_id": "T1", "task_key": "K1", "stage": "deep_search", "status": "failed",
                  "planned_relative_cost": 2, "model_version": "m1"}
        first = restart_failed_task({"tasks": [failed]}, "T1", budget_limits=default_budget_rules(),
                                    config_version="c1")
        self.assertEqual(first["status"], "prepared")
        self.assertEqual(first["task"]["retry_of"], "T1")
        repeated = restart_failed_task(first["state"], "T1", budget_limits=default_budget_rules(),
                                       config_version="c1")
        self.assertEqual(repeated["status"], "already_prepared")

    def test_step_runner_rejects_unconfirmed_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"budgets": {"total_relative_cost": 1}}))
            with self.assertRaisesRegex(ValueError, "confirmed"):
                _runtime_config(path)

    def test_initial_model_health_requires_endpoint_and_intermediate(self):
        incomplete = assess_initial_mlip([{"phase_role": "endpoint", "status": "completed",
                                           "error_ev_per_atom": .1}])
        self.assertEqual(incomplete["status"], "insufficient_evidence")
        complete = assess_initial_mlip([
            {"phase_role": "endpoint", "status": "completed", "error_ev_per_atom": .1},
            {"phase_role": "intermediate", "status": "completed", "error_ev_per_atom": .2}])
        self.assertEqual(complete["status"], "passed")

    def test_validation_does_not_require_monotonic_mae_but_blocks_clear_anomaly(self):
        evaluator = lambda model, data: ({"energy_mae": .10, "force_rmse": .2,
                                          "critical_failure_fraction": 0,
                                          "near_hull_ranking_reversals": 0} if model["version"] == "old"
                                         else {"energy_mae": .11, "force_rmse": .2,
                                               "critical_failure_fraction": 0,
                                               "near_hull_ranking_reversals": 0})
        base = {"max_critical_failure_fraction": .5, "max_near_hull_ranking_reversals": 0}
        loose = validate_mlip({"version": "old"}, {"version": "new"}, [{}], evaluator=evaluator,
                              criteria={**base, "max_energy_mae": .2}, validation_data_version="v1")
        self.assertTrue(loose["passed"]); self.assertEqual(loose["benefit_status"], "no_distinguishable_benefit")
        blocked = validate_mlip({"version": "old"}, {"version": "new"}, [{}], evaluator=evaluator,
                                criteria={**base, "max_energy_mae": .105}, validation_data_version="v1")
        self.assertFalse(blocked["passed"])

    def test_candidate_waits_for_separate_activation_then_can_rollback(self):
        state = {"iteration": 1, "active_model_version": "m1", "active_model": {"version": "m1"},
                 "new_dft_records": [{"structure_id": "S1", "status": "completed", "converged": True,
                                      "checks_passed": True, "energy": -1}]}
        handler = create_model_update_handler(
            trainer=lambda **kw: {"version": "m2", "artifact": "mock"},
            validation_evaluator=lambda model, data: {"energy_mae": .2 if model.get("version") == "m1" else .1,
                                                       "force_rmse": .2, "critical_failure_fraction": 0,
                                                       "near_hull_ranking_reversals": 0},
            validation_data_provider=lambda **kw: [{}])
        config = {"mlip": {"version": "m1"}, "mlip_finetune": {
            "training": {"minimum_new_dft_records": 1},
            "validation": {"max_energy_mae": .3, "max_critical_failure_fraction": .5,
                           "max_near_hull_ranking_reversals": 0},
            "activation": {"requires_separate_approval": True}}}
        candidate = handler(trigger={"action": "RETRAIN_MLIP"}, state=state, manager=None, config=config)
        self.assertEqual(candidate["status"], "awaiting_activation_approval")
        self.assertEqual(candidate["state"]["active_model_version"], "m1")
        version = next(iter(candidate["state"]["candidate_models"]))
        rejected = handler(trigger={"action": "REJECT_CANDIDATE_MODEL",
            "candidate_model_version": version, "user_rejection_reason": "no clear benefit"},
            state=candidate["state"], manager=None, config=config)
        self.assertEqual(rejected["status"], "candidate_rejected_by_user")
        self.assertEqual(rejected["state"]["active_model_version"], "m1")
        activated = handler(trigger={"action": "ACTIVATE_CANDIDATE_MODEL",
            "candidate_model_version": version, "user_approval_reason": "mock accepted"},
            state=candidate["state"], manager=None, config=config)
        self.assertEqual(activated["status"], "activated")
        anomaly = assess_model_refresh(activated["state"], {
            "failure_fraction": .8, "near_hull_ranking_reversals": 0},
            limits={"max_failure_fraction": .5, "max_near_hull_ranking_reversals": 0})
        self.assertEqual(anomaly["status"], "paused_anomaly")
        rolled = rollback_model(anomaly["state"], target_version="m1", user_approved=True,
                                reason="critical refresh failures")
        self.assertEqual(rolled["status"], "rolled_back_paused")
        self.assertEqual(rolled["state"]["active_model_version"], "m1")


if __name__ == "__main__": unittest.main()

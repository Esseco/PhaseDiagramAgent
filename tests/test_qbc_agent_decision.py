import unittest

from config_layer.defaults.default_dft_decision_config import default_dft_decision_config
from execution_layer.workflows.run_qbc_dft_decision_flow import run_qbc_dft_decision_flow
from execution_layer.dispatch.create_tool_registry import create_tool_registry
from config_layer.defaults.default_layered_search_config import default_layered_search_config


class QbcAgentDecisionTest(unittest.TestCase):
    def candidates(self):
        return [{"candidate_id": "near", "branch_id": "B1", "qbc": {"status": "completed", "f_std_max": 1., "f_std_p95": .8, "energy_std": .2}, "predicted_Ehull": .01, "estimated_cost": 1}, {"candidate_id": "far", "branch_id": "B2", "qbc": {"status": "completed", "f_std_max": 1., "f_std_p95": .8, "energy_std": .2}, "predicted_Ehull": 1., "estimated_cost": 1}, {"candidate_id": "dup", "branch_id": "B1", "duplicate_of": "near", "qbc": {"status": "completed", "f_std_max": .1, "f_std_p95": .1, "energy_std": .01}, "predicted_Ehull": .02, "estimated_cost": 1}]

    def test_agent_cases_retrain_and_fixed_baseline(self):
        candidates = self.candidates()
        scheduler = lambda _: {
            "decisions": [
                {"candidate_id": "near", "action": "DFT_RELAX"},
                {"candidate_id": "far", "action": "DEFER"},
                {"candidate_id": "dup", "action": "DFT_SINGLE_POINT"},
            ],
            "global_action": "RETRAIN_MLIP",
        }
        submitter = lambda **_: {
            "status": "completed", "converged": True,
            "checks_passed": True, "energy": -1.0,
        }
        config = default_dft_decision_config()
        config["retrain"]["minimum_new_dft_records"] = 1
        context = {"remaining_dft_budget": 2000}
        agent = run_qbc_dft_decision_flow(
            candidates, {}, config=config, config_version="c1",
            dft_parameters={"encut": 520}, context=context,
            agent_client=scheduler, dft_submitter=submitter,
            invocation_id="agent",
        )
        fixed = default_dft_decision_config()
        fixed["mode"] = "qbc_fixed"
        fixed["fixed"]["batch_size"] = 1
        baseline = run_qbc_dft_decision_flow(
            candidates, {}, config=fixed, config_version="c1",
            dft_parameters={"encut": 520}, context=context,
            dft_submitter=submitter, invocation_id="fixed",
        )
        self.assertEqual(agent["status"], "rejected")
        self.assertEqual(agent["submissions"], [])
        actions = {item["candidate_id"]: item["action"] for item in agent["validation"]["accepted"]}
        self.assertEqual(actions["near"], "DFT_RELAX")
        self.assertEqual(actions["far"], "DEFER")
        self.assertNotIn("dup", actions)
        duplicate = next(row for row in agent["validation"]["rejected"] if row["candidate_id"] == "dup")
        self.assertEqual(duplicate["action"], "DFT_SINGLE_POINT")
        self.assertEqual(duplicate["reason"], "duplicate_safety_rule")
        self.assertIsNone(agent["retrain"])
        self.assertEqual(baseline["status"], "completed")

    def test_illegal_numeric_and_budget_are_rejected(self):
        config = default_dft_decision_config(); context = {"remaining_dft_budget": 0}
        illegal = lambda _: {"decisions": [{"candidate_id": "near", "action": "DFT_RELAX", "energy_std": 9.}], "global_action": "CONTINUE_DATA_COLLECTION"}
        result = run_qbc_dft_decision_flow(self.candidates(), {}, config=config, config_version="c1", dft_parameters={"encut": 520}, context=context, agent_client=illegal, invocation_id="x")
        self.assertEqual(result["status"], "rejected"); self.assertIn("agent_supplied_scientific_numeric_value", result["validation"]["errors"])
        legal = lambda _: {"decisions": [{"candidate_id": "near", "action": "DFT_RELAX"}], "global_action": "CONTINUE_DATA_COLLECTION"}
        budget = run_qbc_dft_decision_flow(self.candidates(), {}, config=config, config_version="c1", dft_parameters={"encut": 520}, context=context, agent_client=legal, invocation_id="b")
        self.assertEqual(budget["validation"]["rejected"][0]["reason"], "dft_budget")
        parameter_override = lambda _: {"decisions": [{"candidate_id": "near", "action": "DFT_RELAX", "dft_parameters": {"encut": 300}}]}
        overridden = run_qbc_dft_decision_flow(self.candidates(), {}, config=config, config_version="c1", dft_parameters={"encut": 520}, context={"remaining_dft_budget": 2000}, agent_client=parameter_override, invocation_id="p")
        self.assertEqual(overridden["status"], "rejected")
        self.assertTrue(any("agent_decision_fields_forbidden" in item for item in overridden["validation"]["errors"]))

    def test_invocation_is_idempotent(self):
        config = default_dft_decision_config(); context = {"remaining_dft_budget": 2000}
        agent = lambda _: {"decisions": [{"candidate_id": "near", "action": "DFT_SINGLE_POINT"}]}
        first = run_qbc_dft_decision_flow(self.candidates(), {}, config=config, config_version="c1", dft_parameters={}, context=context, agent_client=agent, invocation_id="same")
        second = run_qbc_dft_decision_flow(self.candidates(), first["state"], config=config, config_version="c1", dft_parameters={}, context=context, agent_client=lambda _: (_ for _ in ()).throw(RuntimeError()), invocation_id="same")
        self.assertTrue(second["idempotent_replay"])

    def test_pending_dft_keeps_shared_reservation_until_terminal_result(self):
        config = default_dft_decision_config()
        context = {"remaining_dft_budget": 2000}
        agent = lambda _: {"decisions": [{"candidate_id": "near", "action": "DFT_SINGLE_POINT"}]}
        pending = run_qbc_dft_decision_flow(
            self.candidates(), {}, config=config, config_version="c1",
            dft_parameters={}, context=context, agent_client=agent,
            dft_submitter=lambda **_: {"status": "pending", "task_id": "T1"},
            invocation_id="pending-dft",
        )
        task_key = pending["validation"]["accepted"][0]["task_key"]
        self.assertEqual(pending["state"]["budget_reservations"][task_key]["status"], "submitted")
        self.assertEqual(pending["state"].get("used_dft_cost", 0), 0)
        self.assertGreater(pending["state"]["reserved_relative_cost"], 0)

    def test_interactive_policy_waits_then_approves(self):
        config = default_dft_decision_config()
        context = {"remaining_dft_budget": 2000}
        calls = []
        agent = lambda _: {"decisions": [{"candidate_id": "near", "action": "DFT_SINGLE_POINT"}]}
        submitter = lambda **kwargs: calls.append(kwargs) or {"status": "pending"}
        first = run_qbc_dft_decision_flow(
            self.candidates(), {}, config=config, config_version="c1",
            dft_parameters={"encut": 520}, context=context, agent_client=agent,
            dft_submitter=submitter, invocation_id="interactive",
            execution_mode="interactive",
        )
        self.assertEqual(first["status"], "awaiting_approval")
        self.assertEqual(calls, [])
        self.assertIsNone(first["validation"])
        second = run_qbc_dft_decision_flow(
            self.candidates(), first["state"], config=config, config_version="c1",
            dft_parameters={"encut": 520}, context=context,
            agent_client=lambda _: self.fail("stored proposal should be reused"),
            dft_submitter=submitter, invocation_id="interactive",
            execution_mode="interactive", human_feedback="approve",
        )
        self.assertEqual(second["status"], "completed")
        self.assertEqual(len(calls), 1)
        self.assertEqual(second["state"]["action_records"][0]["human_feedback"]["decision"], "approve")

    def test_interactive_policy_can_reject_without_validation(self):
        config = default_dft_decision_config()
        first = run_qbc_dft_decision_flow(
            self.candidates(), {}, config=config, config_version="c1",
            dft_parameters={}, context={"remaining_dft_budget": 2000},
            agent_client=lambda _: {"decisions": [{"candidate_id": "near", "action": "DFT_RELAX"}]},
            invocation_id="reject", execution_mode="interactive",
        )
        rejected = run_qbc_dft_decision_flow(
            self.candidates(), first["state"], config=config, config_version="c1",
            dft_parameters={}, context={"remaining_dft_budget": 2000},
            invocation_id="reject", execution_mode="interactive", human_feedback="reject",
        )
        self.assertEqual(rejected["status"], "rejected_by_user")
        self.assertEqual(rejected["submissions"], [])
        self.assertIsNone(rejected["validation"])

    def test_dry_run_validates_without_reserving_or_submitting(self):
        calls = []
        result = run_qbc_dft_decision_flow(
            self.candidates(), {}, config=default_dft_decision_config(),
            config_version="c1", dft_parameters={},
            context={"remaining_dft_budget": 2000},
            agent_client=lambda _: {"decisions": [{"candidate_id": "near", "action": "DFT_RELAX"}]},
            dft_submitter=lambda **kwargs: calls.append(kwargs),
            invocation_id="dry", execution_mode="dry_run",
        )
        self.assertEqual(result["status"], "planned_only")
        self.assertTrue(result["validation"]["valid"])
        self.assertEqual(result["submissions"], [])
        self.assertEqual(result["state"].get("budget_reservations"), None)
        self.assertEqual(calls, [])

    def test_agent_mode_never_silently_falls_back_to_qbc_rules(self):
        result = run_qbc_dft_decision_flow(
            self.candidates(), {}, config=default_dft_decision_config(),
            config_version="c1", dft_parameters={},
            context={"remaining_dft_budget": 2000}, invocation_id="no-agent",
        )
        self.assertEqual(result["status"], "not_configured")
        self.assertEqual(result["submissions"], [])
        self.assertEqual(result["proposal"]["source"], "none")

    def test_agent_dft_action_passes_common_tool_permission_validation(self):
        config = default_dft_decision_config()
        platform = default_layered_search_config()
        session = {"status": "confirmed", "confirmed_snapshot": {"config_version": "c1", "config": platform}}
        result = run_qbc_dft_decision_flow(
            self.candidates(), {"confirmed_config_version": "c1"},
            config=config, config_version="c1", dft_parameters={},
            context={"remaining_dft_budget": 2000},
            agent_client=lambda _: {"decisions": [{"candidate_id": "near", "action": "DFT_SINGLE_POINT"}]},
            invocation_id="permission", execution_session=session,
            tool_registry=create_tool_registry({"select_dft_candidates": lambda **kwargs: None}), execution_mode="dry_run",
        )
        self.assertEqual(result["status"], "planned_only")
        self.assertTrue(result["tool_validation"]["valid"])


if __name__ == "__main__": unittest.main()

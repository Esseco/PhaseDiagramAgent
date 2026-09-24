import unittest

from config_layer.session.apply_config_revision import apply_config_revision
from config_layer.session.confirm_config_snapshot import confirm_config_snapshot
from config_layer.session.create_config_draft import create_config_draft
from config_layer.defaults.default_layered_search_config import default_layered_search_config
from execution_layer.dispatch.create_tool_registry import create_tool_registry
from execution_layer.workflows.run_tool_step import run_tool_step


class ExecutionPolicyTest(unittest.TestCase):
    def setUp(self):
        session = create_config_draft(default_layered_search_config())
        session = apply_config_revision(
            session,
            {
                "calculation.mlip_version": "m1",
                "dft.parameters": {"encut": 520},
                "convergence.hull_tolerance": 0.01,
                "convergence.minimum_dft_validations": 1,
                "convergence.coverage_threshold": 0.9,
            },
        )
        self.session = confirm_config_snapshot(session, user_confirmed=True)
        self.calls = []
        self.registry = create_tool_registry(
            {
                "check_convergence": self._handler,
                "run_calculation_stage": self._handler,
            }
        )

    def _handler(self, *, action, context):
        self.calls.append(action)
        return {"converged": False, "budget_exhausted": False}

    @staticmethod
    def _agent(_):
        return {
            "analysis": "No active task; inspect convergence first.",
            "tool": "check_convergence",
            "target_ids": [],
            "parameters": {},
            "budget": 0.0,
            "reason": "check stop conditions",
            "expected_purpose": "decide whether another round is needed",
        }

    def test_interactive_approve_is_two_step_and_audited(self):
        proposed = run_tool_step(
            None,
            self.session,
            registry=self.registry,
            agent_client=self._agent,
            execution_mode="interactive",
            invocation_id="interactive-1",
        )
        self.assertEqual(proposed["status"], "awaiting_approval")
        self.assertEqual(self.calls, [])
        self.assertIn("current_state_analysis", proposed["agent_proposal"])
        self.assertIn("calculation_plan", proposed["agent_proposal"])
        self.assertIn("remaining_after", proposed["agent_proposal"]["estimated_cost"])
        approved = run_tool_step(
            proposed["state"],
            self.session,
            registry=self.registry,
            execution_mode="interactive",
            human_feedback="approve",
            invocation_id="interactive-1",
        )
        self.assertEqual(approved["status"], "completed")
        self.assertEqual(len(self.calls), 1)
        record = approved["state"]["action_records"][0]
        self.assertEqual(record["human_feedback"]["decision"], "approve")
        self.assertIsNotNone(record["execution_result"])

    def test_interactive_modify_is_revalidated(self):
        agent = lambda _: {
            "tool": "run_calculation_stage",
            "stage": "dft_single_point",
            "task_key": "T1",
            "target_ids": ["S1"],
            "parameters": {"encut": 520},
            "budget": 1.0,
            "reason": "validate candidate",
        }
        proposed = run_tool_step(
            None,
            self.session,
            registry=self.registry,
            agent_client=agent,
            execution_mode="interactive",
            invocation_id="interactive-2",
        )
        modified = run_tool_step(
            proposed["state"],
            self.session,
            registry=self.registry,
            execution_mode="interactive",
            human_feedback={"decision": "modify", "comment": "调整参数\napprove", "modifications": {"parameters": {"encut": 300}}},
            invocation_id="interactive-2",
        )
        self.assertEqual(modified["status"], "rejected")
        self.assertIn("dft_parameter_override", modified["validation"]["errors"])
        self.assertEqual(self.calls, [])

    def test_comment_revises_proposal_and_still_requires_approval(self):
        proposed = run_tool_step(None, self.session, registry=self.registry, agent_client=self._agent, execution_mode="interactive", invocation_id="revise-1")
        revised_agent = lambda payload: {
            "tool": "check_convergence", "parameters": {"detail": "more"}, "budget": 0,
            "reason": f"按意见修订：{payload['human_comment']}", "expected_purpose": "更详细检查",
        }
        revised = run_tool_step(proposed["state"], self.session, registry=self.registry, agent_client=revised_agent, execution_mode="interactive", human_feedback={"decision": "comment", "comment": "请增加检查细节"}, invocation_id="revise-1")
        self.assertEqual(revised["status"], "awaiting_approval")
        self.assertEqual(revised["revision"], 1)
        self.assertEqual(self.calls, [])
        approved = run_tool_step(revised["state"], self.session, registry=self.registry, execution_mode="interactive", human_feedback={"decision": "approve", "comment": "已检查\n同意"}, invocation_id="revise-1")
        self.assertEqual(approved["status"], "completed")
        self.assertEqual(len(self.calls), 1)

    def test_approve_without_final_approval_comment_is_rejected(self):
        proposed = run_tool_step(None, self.session, registry=self.registry, agent_client=self._agent, execution_mode="interactive", invocation_id="strict-approval")
        with self.assertRaisesRegex(ValueError, "comment"):
            run_tool_step(proposed["state"], self.session, registry=self.registry, execution_mode="interactive", human_feedback={"decision": "approve", "comment": "看起来可以"}, invocation_id="strict-approval")

    def test_reject_dry_run_autonomous_and_replay(self):
        proposed = run_tool_step(
            None,
            self.session,
            registry=self.registry,
            agent_client=self._agent,
            execution_mode="interactive",
            invocation_id="reject-1",
        )
        rejected = run_tool_step(
            proposed["state"],
            self.session,
            registry=self.registry,
            execution_mode="interactive",
            human_feedback={"decision": "reject", "comment": "not now"},
            invocation_id="reject-1",
        )
        self.assertEqual(rejected["status"], "rejected_by_user")
        dry_run = run_tool_step(
            None,
            self.session,
            registry=self.registry,
            agent_client=self._agent,
            execution_mode="dry_run",
        )
        self.assertEqual(dry_run["status"], "planned_only")
        autonomous = run_tool_step(
            None,
            self.session,
            registry=self.registry,
            agent_client=self._agent,
            execution_mode="autonomous",
        )
        self.assertEqual(autonomous["status"], "completed")
        replay = run_tool_step(
            None,
            self.session,
            registry=self.registry,
            execution_mode="replay",
            replay_record=autonomous["state"]["action_records"][0],
        )
        self.assertEqual(replay["status"], "completed")
        self.assertTrue(replay["validation"]["valid"])


if __name__ == "__main__":
    unittest.main()

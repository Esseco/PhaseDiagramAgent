import unittest

from analysis_layer.state.build_decision_context import build_decision_context
from decision_layer.agent.propose_tool_action import propose_agent_tool_action
from execution_layer.policy.validate_tool_action import validate_tool_action
from execution_layer.workflows.run_tool_step import _apply_execution_result
from run.agent_api import _drop_finished_pending, format_workflow_reply


class AgentRetryGuardTest(unittest.TestCase):
    def test_action_record_is_not_a_retryable_task(self):
        session = {"status": "confirmed", "confirmed_snapshot": {
            "config_version": "config-1", "config": {
                "agent": {"allowed_tools": ["restart_failed_task"]},
                "budgets": {"total_relative_cost": 100}, "frozen_parameters": [],
            },
        }}
        state = {"tasks": [], "action_records": [{"record_id": "webui-old", "status": "failed"}]}
        action = {"tool": "restart_failed_task", "task_key": "retry-1",
                  "target_ids": ["webui-old"], "parameters": {}, "budget": 0}
        validation = validate_tool_action(action, state, session,
                                          {"restart_failed_task": {"handler": lambda **_: None}})
        self.assertIn("retry_target_not_failed_task", validation["errors"])

    def test_agent_sees_compact_failure_and_can_retry_generation(self):
        state = {"action_records": [{
            "record_id": "webui-old", "status": "failed",
            "final_action": {"tool": "generate_branches", "decision_context": {"huge": "x" * 10000}},
            "execution_result": {"status": "failed", "error": "generation failed"},
        }], "tasks": []}
        context = build_decision_context(state)
        self.assertEqual(context["retryable_tasks"], [])
        self.assertEqual(context["recent_experience"]["actions"][0]["failure_reason"],
                         "generation failed")
        self.assertNotIn("huge", str(context))
        proposal = propose_agent_tool_action(
            {"config_version": "config-1", "decision_context": context,
             "search_history": [{"action_type": "generate_branches", "status": "failed"}]},
            allowed_tools=["generate_branches"], agent_client=None,
        )
        self.assertEqual(proposal["tool"], "generate_branches")

    def test_execution_rejection_shows_actual_reason(self):
        reply = format_workflow_reply({
            "status": "rejected", "validation": {"valid": True, "errors": []},
            "execution": {"status": "completed", "result": {
                "status": "rejected", "reason": "task_not_found"}},
        }, "state.json")
        self.assertIn("目标计算任务不存在", reply)
        self.assertNotIn("动作校验未通过", reply)

    def test_stale_approval_is_removed_without_touching_a_live_approval(self):
        state = {
            "pending_execution_policies": {
                "old": {"record_id": "old"}, "live": {"record_id": "live"}},
            "action_records": [{"record_id": "old", "status": "rejected"},
                               {"record_id": "live", "status": "awaiting_approval"}],
        }
        updated, changed = _drop_finished_pending(state)
        self.assertTrue(changed)
        self.assertEqual(list(updated["pending_execution_policies"]), ["live"])
        self.assertEqual(len(state["pending_execution_policies"]), 2)

    def test_invalid_agent_retry_falls_back_to_branch_generation(self):
        state = {"config_version": "config-1", "decision_context": {"retryable_tasks": []},
                 "search_history": [{"action_type": "generate_branches", "status": "failed"}]}
        proposal = propose_agent_tool_action(
            state, allowed_tools=["generate_branches", "restart_failed_task"],
            agent_client=lambda _: {"tool": "restart_failed_task", "target_ids": ["webui-old"],
                                    "parameters": {}, "budget": 0},
        )
        self.assertEqual(proposal["tool"], "generate_branches")
        self.assertIn("retryable scientific task_id", proposal["fallback_reason"])

    def test_handler_state_cannot_restore_consumed_approval(self):
        state, status = _apply_execution_result(
            {"pending_execution_policies": {}}, {"tool": "restart_failed_task"},
            {"status": "completed", "result": {"status": "rejected", "reason": "task_not_found",
             "state": {"pending_execution_policies": {"old": {"record_id": "old"}}}}},
            record_id="old", formal=False,
        )
        self.assertEqual(status, "rejected")
        self.assertEqual(state["pending_execution_policies"], {})


if __name__ == "__main__":
    unittest.main()

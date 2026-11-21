"""The execution gates preserve live work while allowing verified local inputs."""
from unittest.mock import patch

from phase_agent.tools.workflows.manual_training_coexistence import blocking_tasks
from phase_agent.tools.workflows.prepare_remote_finetune import prepare_remote_finetune


def test_scoped_frozen_training_does_not_block_verified_old_inputs():
    state = {"tasks": [{"task_id": "old", "status": "pending", "stage": "dft_single_point"}]}
    action = {"tool": "update_mlip", "parameters": {"action": "RETRAIN_MLIP", "prepare_inputs_only": True}, "_decision_team": {"backend": "codex_native"}, "_simulation_plan": {"steps": [1]}}
    with patch("phase_agent.tools.workflows.manual_training_coexistence.verified_ids", return_value={"old"}):
        assert blocking_tasks(state, action) == []
        assert blocking_tasks(state, None)
        assert blocking_tasks(state, {**action, "parameters": {"action": "RETRAIN_MLIP"}})
        assert prepare_remote_finetune(state, {})["status"] == "tasks_awaiting_results"
        assert prepare_remote_finetune(state, {}, training_action=action)["status"] == "not_configured"
    with patch("phase_agent.tools.workflows.manual_training_coexistence.verified_ids", return_value=set()):
        state["tasks"][0]["status"] = "running"
        assert blocking_tasks(state, action)

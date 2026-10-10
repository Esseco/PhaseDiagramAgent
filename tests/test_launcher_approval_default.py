from phase_agent.runtime.local_project_launcher import DEFAULT_PROJECT_SETTINGS
from phase_agent.tools.policy.execution_policy import apply_execution_policy


def test_new_project_requires_approval_before_work():
    assert DEFAULT_PROJECT_SETTINGS["execution_mode"] == "debug"
    assert DEFAULT_PROJECT_SETTINGS["run_steps_per_click"] == 1
    result = apply_execution_policy(
        {"raw_action": {"tool": "generate_branches", "parameters": {}}},
        execution_mode="interactive")
    assert result["status"] == "awaiting_approval"
    assert result["execute"] is False

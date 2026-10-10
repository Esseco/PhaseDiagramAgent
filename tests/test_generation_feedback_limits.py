"""Explicit human generation limits must survive Agent proposal revision."""

from phase_agent.decisions.agent.propose_tool_action import _apply_generation_defaults
from phase_agent.decisions.agent.resolve_explicit_generation_request import (
    _requested_branch_batch_size, _requested_max_det_H,
    resolve_explicit_generation_request,
)
from phase_agent.decisions.agent.revise_tool_proposal import revise_tool_proposal


def test_selected_branch_count_is_not_candidate_quota():
    message = "det(H) ≤ 12，入选上限：200branch"
    assert _requested_max_det_H(message) == 12
    assert _requested_branch_batch_size(message) == 200
    assert _requested_branch_batch_size("生成配额 coverage 300") is None
    assert _requested_max_det_H("超胞上限12，入选上限500branch") == 12


def test_existing_proposal_is_revised_without_llm_or_config_change():
    original = {"tool": "generate_branches", "task_key": "old", "target_ids": [],
                "parameters": {"total_quota": 300, "batch_size": 96,
                               "quotas": {"coverage": 300}}, "budget": 0}
    revised = revise_tool_proposal(
        {"raw_action": original}, "det(H) ≤ 12，入选上限：200branch",
        state={}, allowed_tools=["generate_branches"], agent_client=None,
    )["action"]
    applied = _apply_generation_defaults(
        revised, {}, {"run": {"total_quota": 300, "batch_size": 96,
                                "initial_states_per_branch": 3}},
    )
    assert applied["parameters"]["max_det_H"] == 12
    assert applied["parameters"]["batch_size"] == 200
    assert original["parameters"]["batch_size"] == 96
    assert applied["task_key"] != original["task_key"]


def test_explicit_generation_request_preserves_selected_branch_limit():
    action = resolve_explicit_generation_request(
        "det(H) ≤ 12，入选上限200branch，重新生成branch",
        allowed_tools=["generate_branches"],
    )
    assert action["parameters"] == {"max_det_H": 12, "batch_size": 200}


def test_over_limit_feedback_cancels_old_approvable_proposal():
    from phase_agent.configuration.defaults.default_layered_search_config import default_layered_search_config
    from phase_agent.tools.policy.execution_policy import build_agent_proposal
    from phase_agent.tools.workflows.run_tool_step import run_tool_step

    config = default_layered_search_config()
    state = {"confirmed_config_version": "test-v1", "confirmed_config": config,
             "pending_execution_policies": {}, "action_records": []}
    action = {"tool": "generate_branches", "task_key": "old", "target_ids": [],
              "parameters": {"total_quota": 300, "batch_size": 96,
                             "quotas": {"coverage": 300}}, "budget": 0,
              "reason": "old"}
    state["pending_execution_policies"]["test-plan"] = {
        "record_id": "test-plan", "agent_proposal": build_agent_proposal(action, {}),
    }
    session = {"status": "confirmed", "confirmed_snapshot": {
        "config_version": "test-v1", "config": config}}
    result = run_tool_step(
        state, session, registry={}, execution_mode="interactive",
        invocation_id="test-plan", human_feedback={"decision": "comment",
            "comment": "超胞上限12，入选上限500branch"},
    )
    assert result["status"] == "configuration_revision_required"
    assert result["state"]["pending_execution_policies"] == {}
    assert "500" in result["reason"]


def test_populated_run_does_not_use_new_config_without_rebinding():
    from phase_agent.configuration.defaults.default_layered_search_config import default_layered_search_config
    from phase_agent.tools.workflows.run_tool_step import run_tool_step

    config = default_layered_search_config()
    state = {"confirmed_config_version": "old", "tasks": [{"task_id": "task-1"}],
             "pending_execution_policies": {}, "action_records": []}
    session = {"status": "confirmed", "confirmed_snapshot": {
        "config_version": "new", "config": config}}
    result = run_tool_step(state, session, registry={}, execution_mode="interactive")
    assert result["status"] == "configuration_version_mismatch"
    assert result["state"]["confirmed_config_version"] == "old"


def test_confirmed_mc_policy_revision_rebinds_only_idle_run():
    from copy import deepcopy
    from phase_agent.configuration.defaults.default_layered_search_config import default_layered_search_config
    from phase_agent.configuration.runtime.authorize_generation_policy_revision import authorize_generation_policy_revision

    old = default_layered_search_config()
    new = deepcopy(old)
    new["round_strategy"]["maximum_mc_budget"] = 5000
    new["round_strategy"]["rule_default"]["mc_budget"] = 5000
    state = {"confirmed_config_version": "old", "confirmed_config": old,
             "tasks": [{"task_id": "finished", "status": "completed", "config_version": "old"}]}
    snapshot = {"config_version": "new", "config": new}
    result = authorize_generation_policy_revision(state, snapshot)
    assert result["status"] == "rebound"
    assert result["state"]["confirmed_config_version"] == "new"
    assert result["state"]["tasks"][0]["config_version"] == "old"
    assert authorize_generation_policy_revision(result["state"], snapshot)["status"] == "unchanged"

    active = deepcopy(state)
    active["tasks"].append({"task_id": "running", "status": "running"})
    assert authorize_generation_policy_revision(active, snapshot)["status"] == "active_work"

    scientific = deepcopy(new)
    scientific["mlip"]["name"] = "different-model"
    assert authorize_generation_policy_revision(state, {"config_version": "other",
                                                  "config": scientific})["status"] == "rejected_scientific_change"

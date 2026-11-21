import pytest
from phase_agent.agents.common.action_contracts import action_contract_errors
from phase_agent.agents.strategic.propose_tool_action import propose_agent_tool_action


@pytest.mark.parametrize("change", [{"budget": True}, {"budget": "30"},
    {"budget": {"value": 30}}, {"budget": float("nan")}, {"budget": -1},
    {"parameters": []}, {"target_ids": "S-1"}, {"target_ids": [1]},
    {"evidence_refs": [{}]}, {"reason": 1}, {"action_type": "update_mlip"}])
def test_invalid_wire_fields_rejected(change):
    assert action_contract_errors({"tool": "pause_search", **change})


def test_legacy_alias_and_metadata_are_not_rewritten():
    action = {"action_type": "pause_search", "_llm_usage": {"calls": 1}}
    assert not action_contract_errors(action)
    assert "tool" not in action


def test_invalid_budget_not_silently_accepted_as_zero():
    result = propose_agent_tool_action({}, allowed_tools=["pause_search"],
        agent_client=lambda _: {"tool": "pause_search", "budget": "30"})
    assert result["decision_source"] == "rule"
    assert "Invalid LLM action" in result["fallback_reason"]


def test_dft_preserves_frozen_branch_review_transport_fields(monkeypatch):
    from copy import deepcopy
    import phase_agent.agents.strategic.propose_tool_action as module
    from phase_agent.agents.simulation.agent import _contains_intent

    action = {"tool": "select_dft_candidates", "task_key": "dft-review",
              "parameters": {"decisions": []}, "target_ids": [], "budget": 0,
              "branch_pool_review": {"existing_branch_groups": [],
                  "default_existing_branch_decision": "defer",
                  "default_existing_branch_reason": "review later",
                  "default_existing_branch_evidence_refs": []}}
    action["_simulation_intent"] = deepcopy(action)
    monkeypatch.setattr(module, "request_validated_action", lambda client, payload: deepcopy(action))
    state = {"decision_context": {"branch_pool_facts": {"existing_branches": [
        {"branch_id": "B1"}]}}}
    result = module.propose_agent_tool_action(state, agent_client=object(),
        allowed_tools=["select_dft_candidates"], config={})
    assert result["tool"] == "select_dft_candidates"
    assert result["branch_pool_review"] == action["branch_pool_review"]
    assert _contains_intent(result, action["_simulation_intent"])

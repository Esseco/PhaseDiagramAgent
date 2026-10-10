import pytest
from phase_agent.decisions.agent.action_contracts import action_contract_errors
from phase_agent.decisions.agent.propose_tool_action import propose_agent_tool_action


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

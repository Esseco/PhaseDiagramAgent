from unittest.mock import patch
import pytest

from phase_agent.graphs.proposal_graph import request_with_langgraph


def test_proposal_graph_preserves_output_and_calls_model_once():
    action = {"tool": "pause_search", "purpose": "review", "parameters": {}, "budget": 0}
    with patch("phase_agent.decisions.agent.post_dft_review.request_validated_action", return_value=action) as model:
        result = request_with_langgraph(object(), {"allowed_tools": ["pause_search"]})
    assert result == action
    model.assert_called_once()


def test_invalid_tool_is_not_executed_and_preserves_usage():
    action = {"tool": "pause_search", "purpose": "review", "parameters": {}, "budget": 0,
              "_llm_usage": {"total_tokens": 10}}
    with patch("phase_agent.decisions.agent.post_dft_review.request_validated_action", return_value=action):
        with pytest.raises(ValueError, match="not allowed") as failure:
            request_with_langgraph(object(), {"allowed_tools": ["generate_branches"]})
    assert failure.value.llm_usage == {"total_tokens": 10}


def test_generation_quota_mismatch_repairs_once_before_returning():
    from copy import deepcopy
    action = {"tool": "generate_branches", "parameters": {
        "total_quota": 2, "quotas": {"coverage": 3},
        "generation_plan": [{"strategy": "coverage", "quota": 2, "phase": "O3", "reason": "coverage"}]}, "budget": 0}
    repaired = deepcopy(action)
    repaired["parameters"]["quotas"]["coverage"] = 2
    with patch("phase_agent.decisions.agent.post_dft_review.request_validated_action", side_effect=[action, repaired]) as model:
        result = request_with_langgraph(object(), {"allowed_tools": ["generate_branches"]})
    assert result == repaired
    assert model.call_count == 2
    assert model.call_args.args[1]["mode"] == "proposal_contract_repair"
    assert "generation_plan" in model.call_args.args[1]["validation_errors"][0]


def test_generation_quota_repair_failure_stops_without_default_allocation():
    action = {"tool": "generate_branches", "parameters": {
        "total_quota": 2, "quotas": {"coverage": 3},
        "generation_plan": [{"strategy": "coverage", "quota": 2, "phase": "O3", "reason": "coverage"}]}, "budget": 0}
    with patch("phase_agent.decisions.agent.post_dft_review.request_validated_action", return_value=action) as model:
        with pytest.raises(ValueError, match="generation_plan"):
            request_with_langgraph(object(), {"allowed_tools": ["generate_branches"]})
    assert model.call_count == 2

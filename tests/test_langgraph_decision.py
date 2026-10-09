from unittest.mock import patch
import pytest

from decision_layer.agent.langgraph_decision import request_with_langgraph


def test_proposal_graph_preserves_output_and_calls_model_once():
    action = {"tool": "pause_search", "purpose": "review", "parameters": {}, "budget": 0}
    with patch("decision_layer.agent.post_dft_review.request_validated_action", return_value=action) as model:
        result = request_with_langgraph(object(), {"allowed_tools": ["pause_search"]})
    assert result == action
    model.assert_called_once()


def test_invalid_tool_is_not_executed_and_preserves_usage():
    action = {"tool": "pause_search", "purpose": "review", "parameters": {}, "budget": 0,
              "_llm_usage": {"total_tokens": 10}}
    with patch("decision_layer.agent.post_dft_review.request_validated_action", return_value=action):
        with pytest.raises(ValueError, match="not allowed") as failure:
            request_with_langgraph(object(), {"allowed_tools": ["generate_branches"]})
    assert failure.value.llm_usage == {"total_tokens": 10}

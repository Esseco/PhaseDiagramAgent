import pytest
from phase_agent.decisions.agent.decision_backend import request_decision_action, check_decision_backend


def test_default_langgraph_calls_model_once():
    calls = []
    def client(payload):
        calls.append(payload)
        return {"tool": "pause_search"}
    assert request_decision_action(client, {"instruction": "test", "allowed_tools": ["pause_search"]})["tool"] == "pause_search"
    assert len(calls) == 1


def test_unknown_backend_rejected():
    with pytest.raises(ValueError):
        check_decision_backend("guess")
    with pytest.raises(ValueError):
        check_decision_backend("legacy")


def test_explicit_langgraph_does_not_double_request():
    calls = []
    def client(payload):
        calls.append(payload)
        return {"tool": "pause_search", "budget": 0}
    client.decision_backend = "langgraph"
    result = request_decision_action(client, {"instruction": "test", "allowed_tools": ["pause_search"]})
    assert result == {"tool": "pause_search", "budget": 0}
    assert len(calls) == 1

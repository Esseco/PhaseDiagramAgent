"""Explicit decision backend selection, independent of transport clients."""
import importlib.util


def check_decision_backend(backend):
    if backend != "langgraph":
        raise ValueError("decision backend must be langgraph")
    if backend == "langgraph" and importlib.util.find_spec("langgraph") is None:
        raise ValueError("LangGraph dependency unavailable; install requirements.txt")


def request_decision_action(agent_client, payload):
    backend = getattr(agent_client, "decision_backend", "langgraph")
    check_decision_backend(backend)
    if backend == "langgraph":
        from decision_layer.agent.langgraph_decision import request_with_langgraph
        return request_with_langgraph(agent_client, payload)

"""Reusable scientific lifecycle; compilation never processes project data."""
from functools import lru_cache
from contextvars import ContextVar
from contextlib import contextmanager
from orchestration.search_workflow_graph import build_search_workflow_graph
from orchestration.event_loop_graph import build_event_loop_graph
from orchestration.tool_action_graph import build_tool_action_graph

_active_graph = ContextVar("scientific_graph", default=None)


@lru_cache(maxsize=1)
def scientific_graph():
    from execution_layer.dispatch.create_tool_registry import create_tool_registry
    tools = build_tool_action_graph(registry=create_tool_registry(), group_actions=True)
    actions = build_event_loop_graph(tool_graph=tools)
    return build_search_workflow_graph(action_graph=actions)


@contextmanager
def use_scientific_graph(graph):
    """Bind only the current chat invocation, never a process-global handler."""
    token = _active_graph.set(graph)
    try:
        yield
    finally:
        _active_graph.reset(token)


def current_scientific_graph():
    graph = _active_graph.get()
    return scientific_graph() if graph is None else graph

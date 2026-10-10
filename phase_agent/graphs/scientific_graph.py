"""Reusable scientific lifecycle; compilation never processes project data."""

from functools import lru_cache
from contextvars import ContextVar
from contextlib import contextmanager
from phase_agent.graphs.actions.graph import build_tool_action_graph

_active_graph = ContextVar("scientific_graph", default=None)


@lru_cache(maxsize=1)
def scientific_graph():
    from phase_agent.tools.dispatch.create_tool_registry import create_tool_registry

    tools = build_tool_action_graph(registry=create_tool_registry(), group_actions=True)
    from phase_agent.graphs.project.graph import build_react_lifecycle_graph

    return build_react_lifecycle_graph(action_graph=tools)


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

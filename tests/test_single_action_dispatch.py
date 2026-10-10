"""Production single-action boundary, independent of the low-level loop utility."""

import importlib
import pytest
from phase_agent.runtime.main import run_workflow
from phase_agent.graphs.scientific_graph import scientific_graph
from phase_agent.tools.dispatch.create_tool_registry import create_tool_registry
from phase_agent.tools.workflows.create_workflow_handlers import create_workflow_handlers
from tests.test_event_loop import _session


@pytest.mark.parametrize("limit", [0, 2, 3, -1, True, "1", 1.0])
def test_multiple_or_invalid_action_limits_fail_before_any_write(limit, tmp_path):
    state_path = tmp_path / "new-project" / "state.json"
    with pytest.raises(ValueError, match="exactly one action"):
        run_workflow(None, {}, {"state_path": str(state_path)}, {}, max_steps=limit)
    assert not state_path.parent.exists()


def test_single_action_persists_once_without_invoking_loop_graph(monkeypatch, tmp_path):
    module = importlib.import_module("phase_agent.tools.workflows.run_event_loop")

    def forbidden(*args, **kwargs):
        raise AssertionError("Production must not enter a multi-step graph")

    monkeypatch.setattr(module, "run_event_loop_graph", forbidden)
    calls = []
    handlers = create_workflow_handlers()
    state_path = tmp_path / "state.json"
    result = module.run_action_turn(
        {},
        _session(),
        registry=create_tool_registry(handlers),
        agent_client=lambda payload: (
            calls.append(payload)
            or {
                "tool": "pause_search",
                "parameters": {},
                "budget": 0,
                "reason": "pause",
            }
        ),
        execution_mode="autonomous",
        invocation_id="single-action",
        state_path=state_path,
    )
    assert result["status"] == "paused"
    assert result["steps_executed"] == 1
    assert result["state"]["event_index"] == 1
    assert len(calls) == 1 and state_path.is_file()


def test_production_topology_directly_exposes_approval_and_tools():
    graph = scientific_graph()
    children = dict(graph.get_subgraphs(recurse=True))
    assert children["react"].name == "approved_scientific_action"
    assert "approval_gate" in children["react"].nodes
    assert not any(child.name == "bounded_action_iteration" for child in children.values())
    assert not any("execute_validated_action" in path for path in children)

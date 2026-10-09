from orchestration.runtime_context import WorkflowRuntime
import json
from concurrent.futures import ThreadPoolExecutor
import pytest
from orchestration.event_loop_graph import run_event_loop_graph
from execution_layer.dispatch.create_tool_registry import create_tool_registry
from orchestration.search_workflow_graph import build_search_workflow_graph
from orchestration.tool_action_graph import build_tool_action_graph


def test_lifecycle_wait_does_not_reach_decision_or_tools():
    calls = []
    def stage(name):
        def call(frame):
            calls.append(name)
            return frame
        return call
    graph = build_search_workflow_graph(
        initialize=lambda: {"_workflow_prepared": True},
        collect=stage("collect"), analyze=stage("analyze"),
        wait=lambda frame: {"status": "awaiting_manual_submission"},
        assess=stage("assess"), act=stage("act"), finalize=stage("finalize"))
    assert graph.invoke({}, context=WorkflowRuntime())["response"]["status"] == "awaiting_manual_submission"
    assert calls == ["collect", "analyze"]


def test_lifecycle_analysis_and_export_precede_decision():
    calls = []
    def stage(name):
        def call(frame):
            calls.append(name)
            return frame
        return call
    graph = build_search_workflow_graph(
        initialize=lambda: {"_workflow_prepared": True},
        collect=stage("collect"), analyze=stage("analyze"), wait=stage("wait"),
        assess=stage("assess_export"), act=stage("act"),
        finalize=lambda frame: {"status": "done"})
    assert graph.invoke({}, context=WorkflowRuntime())["response"] == {"status": "done"}
    assert calls == ["collect", "analyze", "wait", "assess_export", "act"]


def test_all_registered_tools_have_graph_branches_but_only_selected_runs():
    calls = []
    registry = create_tool_registry()
    graph = build_tool_action_graph(
        registry=registry, initialize=lambda: {"_graph_prepared": True}, refresh=lambda frame: frame,
        prepare=lambda frame: frame, approve=lambda frame: frame,
        validate=lambda frame: {**frame, "action": {"tool": "pause_search"}, "dispatch_ready": True},
        dispatch=lambda frame: calls.append(frame["action"]["tool"]) or frame,
        finish=lambda frame: {"status": "paused"})
    assert all("tool__" + name in graph.get_graph().nodes for name in registry)
    updates = list(graph.stream({}, context=WorkflowRuntime(), stream_mode="updates"))
    assert calls == ["pause_search"]
    tool_nodes = [name for update in updates for name in update if name.startswith("tool__")]
    assert tool_nodes == ["tool__pause_search"]


def test_approval_wait_never_validates_or_executes():
    calls = []
    graph = build_tool_action_graph(
        registry=create_tool_registry(), initialize=lambda: {"_graph_prepared": True},
        refresh=lambda frame: frame, prepare=lambda frame: frame,
        approve=lambda frame: {"status": "awaiting_approval"},
        validate=lambda frame: calls.append("validate"),
        dispatch=lambda frame: calls.append("dispatch"), finish=lambda frame: calls.append("finish"))
    assert graph.invoke({}, context=WorkflowRuntime())["response"]["status"] == "awaiting_approval"
    assert calls == []


def test_failed_validation_goes_to_audit_without_tool_execution():
    calls = []
    graph = build_tool_action_graph(
        registry=create_tool_registry(), initialize=lambda: {"_graph_prepared": True},
        refresh=lambda frame: frame, prepare=lambda frame: frame, approve=lambda frame: frame,
        validate=lambda frame: {**frame, "dispatch_ready": False},
        dispatch=lambda frame: calls.append("dispatch"), finish=lambda frame: {"status": "rejected"})
    assert graph.invoke({}, context=WorkflowRuntime())["response"]["status"] == "rejected"
    assert calls == []


def test_runtime_objects_are_not_streamed_and_invocations_are_isolated():
    graph = build_search_workflow_graph(
        initialize=lambda: {"_workflow_prepared": True, "client": object()},
        collect=lambda frame: frame, analyze=lambda frame: frame,
        wait=lambda frame: frame, assess=lambda frame: frame, act=lambda frame: frame,
        finalize=lambda frame: {"status": "done"})

    def invoke_once(_):
        context = WorkflowRuntime()
        updates = list(graph.stream({}, context=context, stream_mode="updates"))
        json.dumps(updates)
        assert all("frame" not in update for item in updates for update in item.values())
        return context.frame["client"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        clients = list(pool.map(invoke_once, range(2)))
    assert clients[0] is not clients[1]


def test_low_level_graph_requires_explicit_runtime_context():
    graph = build_search_workflow_graph(
        initialize=lambda: {"_workflow_prepared": True},
        collect=lambda frame: frame, analyze=lambda frame: frame,
        wait=lambda frame: frame, assess=lambda frame: frame,
        act=lambda frame: frame, finalize=lambda frame: {})
    with pytest.raises(ValueError, match="context=WorkflowRuntime"):
        graph.invoke({})


def test_event_graph_persists_each_step_and_stops_at_bound():
    calls = []

    def execute(state):
        calls.append(("execute", state["offset"]))
        return {"step_id": str(state["offset"]), "response": {"status": "completed"}}

    def persist(state):
        calls.append(("persist", state["offset"]))
        return {"offset": state["offset"] + 1}

    result = run_event_loop_graph(
        max_steps=2, execute=execute, persist=persist,
        route=lambda state: "__end__" if state["offset"] >= 2 else "execute_validated_action")
    assert result["offset"] == 2
    assert calls == [("execute", 0), ("persist", 0), ("execute", 1), ("persist", 1)]


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_event_graph_rejects_invalid_bound_before_any_callback(limit):
    def forbidden(state):
        raise AssertionError("Must not execute")

    with pytest.raises(ValueError, match="正整数"):
        run_event_loop_graph(max_steps=limit, execute=forbidden, persist=forbidden, route=forbidden)

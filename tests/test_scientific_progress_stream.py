from phase_agent.graphs.project.graph import build_react_lifecycle_graph as build_search_workflow_graph
from phase_agent.graphs.runtime_context import WorkflowRuntime


def test_native_progress_stream_reports_stage_before_completion():
    calls = []
    def stage(frame):
        calls.append("ran")
        return frame
    graph = build_search_workflow_graph(initialize=lambda: {"_workflow_prepared": True},
        collect=stage, analyze=stage, wait=stage, assess=stage,
        act=lambda frame, **kwargs: frame, finalize=lambda frame: {"status": "done"})
    events = [event for _, event in graph.stream({}, context=WorkflowRuntime(), stream_mode="custom", subgraphs=True)]
    assert events[0] == {"kind": "scientific_progress", "node": "initialize_confirmed_run",
                         "label": "核对项目配置", "status": "running"}
    assert any(event["node"] == "edge_direction_review" and event["status"] == "running" for event in events)
    assert events[-1]["node"] == "prepare_inputs_and_finalize"
    assert events[-1]["status"] == "completed"
    assert len(calls) == 4


def test_waiting_stage_is_not_reported_as_executed():
    graph = build_search_workflow_graph(initialize=lambda: {"_workflow_prepared": True},
        collect=lambda frame: frame, analyze=lambda frame: frame,
        wait=lambda frame: {"status": "awaiting_approval"},
        assess=lambda frame: (_ for _ in ()).throw(AssertionError("must not run")))
    events = [event for _, event in graph.stream({}, context=WorkflowRuntime(), stream_mode="custom", subgraphs=True)]
    assert events[-1]["node"] == "results_wait_gate"
    assert events[-1]["status"] == "waiting"

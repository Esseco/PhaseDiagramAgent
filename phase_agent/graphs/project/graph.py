"""Readable project lifecycle; observation and approved tools are reusable subgraphs."""

from phase_agent.graphs.state import SearchWorkflowState
from phase_agent.graphs.runtime_context import WorkflowRuntime
from phase_agent.graphs.project.nodes import create_lifecycle_nodes


def build_react_lifecycle_graph(*, action_graph=None, checkpointer=None, **stages):
    from langgraph.graph import StateGraph, START, END
    from langgraph.types import interrupt, Command

    nodes = create_lifecycle_nodes(action_graph=action_graph, **stages)
    observations = StateGraph(SearchWorkflowState, context_schema=WorkflowRuntime)
    steps = [
        "collect_and_reconcile",
        "batch_recovery",
        "training_lifecycle",
        "edge_direction_review",
        "scientific_feedback",
    ]
    for name in steps:
        observations.add_node(name, nodes[name])
    observations.add_edge(START, steps[0])
    for left, right in zip(steps, steps[1:]):
        observations.add_edge(left, right)
    observations.add_edge(steps[-1], END)

    def after_initialize(state):
        return END if state.get("response") is not None else "observe"

    def is_waiting(state):
        response = state.get("completed_response") or state.get("response") or {}
        from phase_agent.graphs.wait_contract import wait_boundary

        return wait_boundary(response.get("status")) is not None

    def after_wait(state):
        if state.get("response") is None:
            return "analyze"
        return "wait_for_project_input" if state.get("request_id") and is_waiting(state) else END

    def after_finalize(state):
        return "wait_for_project_input" if state.get("request_id") and is_waiting(state) else END

    def wait_for_project_input(state):
        response = state.get("completed_response") or state.get("response") or {}
        from phase_agent.graphs.wait_contract import wait_boundary

        signal = interrupt(
            {
                "kind": "project_wait",
                "status": response.get("status"),
                "request_id": state.get("request_id"),
                "wait_boundary": wait_boundary(response.get("status")),
            }
        )
        if not isinstance(signal, dict) or signal.get("signal") != "reconcile":
            raise ValueError("invalid_project_resume_signal")
        return Command(
            update={
                "response": None,
                "completed_response": None,
                "business_frame": {},
                "request_id": signal["request_id"],
            },
            goto="initialize",
        )

    graph = StateGraph(SearchWorkflowState, context_schema=WorkflowRuntime)
    graph.add_node("initialize", nodes["initialize_confirmed_run"])
    observation = observations.compile(checkpointer=None, name="project_observation")
    # These are the exact read-only graphs called by the existing batch node.
    observation.nodes["batch_recovery"].subgraphs = nodes["batch_recovery"].scientific_children
    graph.add_node("observe", observation)
    graph.add_node("wait", nodes["results_wait_gate"])
    graph.add_node("analyze", nodes["assess_and_export_round"])
    graph.add_node("react", nodes["bounded_action_graph"])
    graph.add_node("finalize", nodes["prepare_inputs_and_finalize"])
    graph.add_node("wait_for_project_input", wait_for_project_input)
    graph.add_edge(START, "initialize")
    graph.add_conditional_edges("initialize", after_initialize, [END, "observe"])
    graph.add_edge("observe", "wait")
    graph.add_conditional_edges("wait", after_wait, [END, "analyze", "wait_for_project_input"])
    graph.add_edge("analyze", "react")
    graph.add_edge("react", "finalize")
    graph.add_conditional_edges("finalize", after_finalize, [END, "wait_for_project_input"])
    from phase_agent.graphs.subgraph_visibility import expose_child

    compiled = graph.compile(checkpointer=checkpointer, name="project_react_workflow")
    expose_child(compiled, "observe", observation)
    expose_child(compiled, "react", action_graph)
    return compiled

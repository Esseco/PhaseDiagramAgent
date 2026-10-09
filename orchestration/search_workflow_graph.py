"""Lifecycle routing; dependencies live in Runtime context, not graph state."""
from orchestration.state import SearchWorkflowState
from orchestration.runtime_context import WorkflowRuntime
from orchestration.lifecycle_nodes import create_lifecycle_nodes


def build_search_workflow_graph(*, initialize=None, collect=None, analyze=None, wait=None,
                                assess=None, act=None, finalize=None, action_graph=None):
    from langgraph.graph import StateGraph, START, END

    nodes = create_lifecycle_nodes(initialize=initialize, collect=collect, analyze=analyze,
        wait=wait, assess=assess, act=act, finalize=finalize, action_graph=action_graph)

    def route_initial(state):
        return END if "response" in state else "collect_and_reconcile"

    def route_wait(state):
        return END if "response" in state else "assess_and_export_round"

    graph = StateGraph(SearchWorkflowState, context_schema=WorkflowRuntime)
    for name, node in nodes.items():
        graph.add_node(name, node)
    graph.add_edge(START, "initialize_confirmed_run")
    graph.add_conditional_edges("initialize_confirmed_run", route_initial, [END, "collect_and_reconcile"])
    graph.add_edge("collect_and_reconcile", "batch_recovery")
    graph.add_edge("batch_recovery", "training_lifecycle")
    graph.add_edge("training_lifecycle", "scientific_feedback")
    graph.add_edge("scientific_feedback", "results_wait_gate")
    graph.add_conditional_edges("results_wait_gate", route_wait, [END, "assess_and_export_round"])
    graph.add_edge("assess_and_export_round", "bounded_action_graph")
    graph.add_edge("bounded_action_graph", "prepare_inputs_and_finalize")
    graph.add_edge("prepare_inputs_and_finalize", END)
    return graph.compile(name="scientific_lifecycle")


def run_search_workflow_graph(*, workflow_graph=None, **stages):
    if workflow_graph is None:
        from orchestration.scientific_graph import current_scientific_graph
        workflow_graph = current_scientific_graph()
    context = WorkflowRuntime(stages=stages, expose_response=False)
    workflow_graph.invoke({}, {"recursion_limit": 32}, context=context)
    if context.response is None:
        raise RuntimeError("Scientific workflow did not return a business response")
    return context.response

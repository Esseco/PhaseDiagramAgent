"""Scientific action grouping; selection remains the validated tool identifier."""

GROUPS = {
    "branch_search": {"generate_branches", "prepare_dedup_batch"},
    "structure_and_batch_inputs": {"select_candidates", "prepare_local_batch_files"},
    "mc_search": {"allocate_mc_bohb"},
    "dft_inputs": {"select_dft_candidates"},
    "mlip_finetune": {"update_mlip"},
    "model_refresh": {"reevaluate_candidates"},
    "convergence_and_control": {
        "check_convergence",
        "pause_search",
        "adjust_strategy",
        "restart_failed_task",
        "run_calculation_stage",
    },
}


def group_for(tool):
    return next(
        (group for group, tools in GROUPS.items() if tool in tools), "convergence_and_control"
    )


def build_action_group(names, dispatch, group):
    from langgraph.graph import StateGraph, START, END
    from phase_agent.graphs.state import ToolActionState
    from phase_agent.graphs.runtime_context import WorkflowRuntime

    graph = StateGraph(ToolActionState, context_schema=WorkflowRuntime)

    def select(state):
        return "tool__" + state["selected_tool"]

    graph.add_conditional_edges(START, select, ["tool__" + name for name in names])
    for name in names:
        graph.add_node("tool__" + name, dispatch)
        graph.add_edge("tool__" + name, END)
    return graph.compile(checkpointer=False, name=group)

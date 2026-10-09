"""Explicit gated tool routing with invocation-local Runtime dependencies."""
from orchestration.state import ToolActionState
from orchestration.runtime_context import WorkflowRuntime
from orchestration.tool_nodes import create_tool_nodes


def build_tool_action_graph(*, registry, initialize=None, refresh=None, prepare=None,
                            approve=None, validate=None, dispatch=None, finish=None,
                            group_actions=False):
    from langgraph.graph import StateGraph, START, END

    nodes = create_tool_nodes(initialize=initialize, refresh=refresh, prepare=prepare, approve=approve, validate=validate, dispatch=dispatch, finish=finish)

    def after_prepare(state):
        return END if "response" in state else "approval_gate"

    def after_initialize(state):
        return END if "response" in state else "model_refresh_barrier"

    def after_refresh(state):
        return END if "response" in state else "prepare_proposal"

    def after_approval(state):
        return END if "response" in state else "validate_action"

    def route_tool(state):
        if group_actions and state["selected_tool"]:
            from orchestration.action_groups import group_for
            return group_for(state["selected_tool"])
        return "tool__" + state["selected_tool"] if state["selected_tool"] else "audit_outcome"

    graph = StateGraph(ToolActionState, context_schema=WorkflowRuntime)
    for name, node in nodes.items():
        if name == "dispatch":
            continue
        graph.add_node(name, node)
    graph.add_edge(START, "initialize_action")
    graph.add_conditional_edges("initialize_action", after_initialize, [END, "model_refresh_barrier"])
    graph.add_conditional_edges("model_refresh_barrier", after_refresh, [END, "prepare_proposal"])
    graph.add_conditional_edges("prepare_proposal", after_prepare, [END, "approval_gate"])
    graph.add_conditional_edges("approval_gate", after_approval, [END, "validate_action"])
    if group_actions:
        from orchestration.action_groups import group_for, build_action_group
        groups = {}
        for name in registry:
            groups.setdefault(group_for(name), []).append(name)
        graph.add_conditional_edges("validate_action", route_tool, ["audit_outcome", *groups])
        for group, names in groups.items():
            graph.add_node(group, build_action_group(names, nodes["dispatch"], group))
            graph.add_edge(group, "audit_outcome")
    else:
        graph.add_conditional_edges("validate_action", route_tool,
                                    ["audit_outcome", *("tool__" + name for name in registry)])
        for name in registry:
            graph.add_node("tool__" + name, nodes["dispatch"])
            graph.add_edge("tool__" + name, "audit_outcome")
    graph.add_edge("audit_outcome", END)
    return graph.compile(name="approved_scientific_action")


def run_tool_action_graph(*, tool_graph=None, registry, **stages):
    # Independent bound: nested graphs otherwise inherit outer recursion limits.
    if tool_graph is None:
        tool_graph = build_tool_action_graph(registry=registry)
    supported = set(tool_graph.nodes)
    for _, child in tool_graph.get_subgraphs(recurse=True):
        supported.update(child.nodes)
    unsupported = [name for name in registry if "tool__" + name not in supported]
    if unsupported:
        raise ValueError(f"Tools absent from compiled action graph: {unsupported}")
    context = WorkflowRuntime(stages=stages, expose_response=False)
    tool_graph.invoke({}, {"recursion_limit": 32}, context=context)
    if context.response is None:
        raise RuntimeError("Tool graph did not return a business response")
    return context.response

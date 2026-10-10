"""One dialogue graph for online, offline and restored sessions."""

from functools import lru_cache

from langgraph.graph import StateGraph, START, END
from phase_agent.graphs.dialogue.state import TurnState, TurnRuntime
from phase_agent.graphs.dialogue.nodes import (
    read_project,
    history,
    route_request,
    configure,
    inspect_configuration,
    review,
    decide,
    present,
    operation_node,
    scientific_workflow,
)


@lru_cache(maxsize=4)
def build_project_turn_graph(science=None):
    from phase_agent.runtime.turn_process import timed_node

    graph = StateGraph(TurnState, context_schema=TurnRuntime)
    for name, node in (
        ("read_project", read_project),
        ("configure", configure),
        ("inspect_configuration", inspect_configuration),
        ("review", review),
        ("decide", decide),
        ("present", present),
    ):
        graph.add_node(name, timed_node(name, node))
    graph.add_node("scientific_workflow", timed_node("scientific_workflow", scientific_workflow))
    import importlib

    maintenance = [
        "inspection",
        "config_import",
        "dft_recovery",
        "export",
        "finetune",
        "mc",
        "rerun",
        "model",
        "migration",
    ]
    preflight = StateGraph(TurnState, context_schema=TurnRuntime)
    for name in maintenance:
        module = importlib.import_module("phase_agent.graphs.dialogue.maintenance." + name)
        preflight.add_node(name, timed_node(name, operation_node(module.handle)))
    preflight.add_node("restore_history", timed_node("restore_history", history))
    preflight.add_node("route_request", timed_node("route_request", route_request))
    preflight.add_edge(START, maintenance[0])
    chain = [*maintenance, "restore_history", "route_request"]
    for left, right in zip(chain, chain[1:]):
        preflight.add_conditional_edges(
            left, lambda s, next_node=right: END if "reply" in s else next_node, [END, right]
        )
    preflight.add_edge("route_request", END)
    graph.add_node("prepare_context", preflight.compile(name="request_context"))
    graph.add_edge(START, "read_project")
    graph.add_conditional_edges(
        "read_project",
        lambda s: END if "reply" in s else "prepare_context",
        [END, "prepare_context"],
    )
    graph.add_conditional_edges(
        "prepare_context",
        lambda s: END if "reply" in s else s["operation"],
        [END, "configure", "review", "decide"],
    )
    graph.add_conditional_edges(
        "decide",
        lambda s: END if "reply" in s else s["operation"],
        {
            END: END,
            "configure": "configure",
            "inspect_configuration": "inspect_configuration",
            "present": "present",
            "scientific_workflow": "scientific_workflow",
        },
    )
    graph.add_edge("scientific_workflow", "present")
    for name in ("configure", "inspect_configuration", "review", "present"):
        graph.add_edge(name, END)
    from phase_agent.graphs.subgraph_visibility import expose_child

    compiled = graph.compile(checkpointer=False, name="project_dialogue")
    if science is not None:
        for name in ("scientific_workflow", "review"):
            expose_child(compiled, name, science)
    return compiled


def run_project_turn(handler, messages, conversation_id):
    from phase_agent.graphs.scientific_graph import current_scientific_graph

    return build_project_turn_graph(current_scientific_graph()).invoke(
        {}, context=TurnRuntime(handler, list(messages), conversation_id)
    )["reply"]

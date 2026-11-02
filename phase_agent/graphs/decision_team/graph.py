"""Bounded three-role reasoning inside the existing proposal/approval boundary."""

from functools import lru_cache
from langgraph.graph import StateGraph, START, END
from phase_agent.graphs.decision_team.state import DecisionTeamState
from phase_agent.graphs.decision_team.runtime import DecisionTeamRuntime, merge_usage
from phase_agent.graphs.decision_team import nodes


@lru_cache(maxsize=1)
def build_proposal_graph():
    graph = StateGraph(DecisionTeamState, context_schema=DecisionTeamRuntime)
    for name in (
        "prepare_evidence",
        "consultation_plan",
        "exploration",
        "reliability",
        "scientific_proposal",
        "coordinated_proposal",
        "validate_proposal",
    ):
        graph.add_node(name, getattr(nodes, name))

    def start_route(state, runtime):
        return (
            "consultation_plan"
            if runtime.context.team_enabled() and not runtime.context.semantic_dialogue()
            else "scientific_proposal"
        )

    def draft_route(state, runtime):
        from phase_agent.agents.common.dialogue_contract import is_dialogue

        return (
            "consultation_plan"
            if runtime.context.team_enabled() and not is_dialogue(state["proposal"])
            else "validate_proposal"
        )

    def expert_route(state):
        roles = state["consultation"]["consult"]
        return (
            "exploration"
            if "exploration" in roles
            else "reliability"
            if "reliability" in roles
            else "coordinated_proposal"
        )

    graph.add_edge(START, "prepare_evidence")
    graph.add_conditional_edges(
        "prepare_evidence", start_route, ["consultation_plan", "scientific_proposal"]
    )
    graph.add_conditional_edges(
        "scientific_proposal", draft_route, ["consultation_plan", "validate_proposal"]
    )
    graph.add_conditional_edges(
        "consultation_plan", expert_route, ["exploration", "reliability", "coordinated_proposal"]
    )
    graph.add_conditional_edges(
        "exploration",
        lambda s: (
            "reliability"
            if "reliability" in s["consultation"]["consult"]
            else "coordinated_proposal"
        ),
        ["reliability", "coordinated_proposal"],
    )
    graph.add_edge("reliability", "coordinated_proposal")
    graph.add_edge("coordinated_proposal", "validate_proposal")
    graph.add_edge("validate_proposal", END)
    return graph.compile(checkpointer=False, name="scientific_decision_team")


def request_with_langgraph(agent_client, payload):
    # Existing callers own their potentially large state/history. Team evidence
    # is projected separately; the legacy proposal path must not copy it again.
    runtime = DecisionTeamRuntime(agent_client, payload)
    try:
        result = build_proposal_graph().invoke({}, context=runtime)["proposal"]
    except Exception as error:
        usage = merge_usage(runtime.usages)
        if usage:
            error.llm_usage = usage
        raise
    usage = merge_usage(runtime.usages)
    if usage:
        result["_llm_usage"] = usage
    return result

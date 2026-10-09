"""Scientific proposal then validation; no tool execution permission."""
from orchestration.state import ProposalState
from orchestration.runtime_context import ProposalRuntime
from orchestration.proposal_nodes import scientific_proposal, validate_proposal


def build_proposal_graph():
    from langgraph.graph import StateGraph, START, END

    graph = StateGraph(ProposalState, context_schema=ProposalRuntime)
    graph.add_node("scientific_proposal", scientific_proposal)
    graph.add_node("validate_proposal", validate_proposal)
    graph.add_edge(START, "scientific_proposal")
    graph.add_edge("scientific_proposal", "validate_proposal")
    graph.add_edge("validate_proposal", END)
    return graph.compile()


def request_with_langgraph(agent_client, payload):
    return build_proposal_graph().invoke(
        {}, context=ProposalRuntime(agent_client, payload))["proposal"]

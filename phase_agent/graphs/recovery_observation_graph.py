"""Expose task recovery and execution auditing as one actual observation child."""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class ObservationState(TypedDict, total=False):
    business: dict
    state_path: str | None
    batch_reports: dict
    execution_report: dict


def build_recovery_observation_graph(batch_graph, recovery_graph):
    def tasks(state):
        from phase_agent.graphs.batch_recovery_graph import recover_batch_stages

        return {
            "batch_reports": recover_batch_stages(
                state["business"], state["state_path"], graph=batch_graph
            )
        }

    def execution(state):
        from phase_agent.graphs.execution_recovery_graph import execution_recovery_report

        report = (
            execution_recovery_report(state["state_path"], state["business"], graph=recovery_graph)
            if state["state_path"]
            else {"status": "clear", "unsettled": [], "checks": []}
        )
        return {"execution_report": report}

    graph = StateGraph(ObservationState)
    graph.add_node("recover_registered_tasks", tasks)
    graph.add_node("audit_interrupted_execution", execution)
    graph.add_edge(START, "recover_registered_tasks")
    graph.add_edge("recover_registered_tasks", "audit_interrupted_execution")
    graph.add_edge("audit_interrupted_execution", END)
    compiled = graph.compile(checkpointer=False, name="recovery_observation")
    from phase_agent.graphs.subgraph_visibility import expose_child

    expose_child(compiled, "recover_registered_tasks", batch_graph)
    expose_child(compiled, "audit_interrupted_execution", recovery_graph)
    return compiled

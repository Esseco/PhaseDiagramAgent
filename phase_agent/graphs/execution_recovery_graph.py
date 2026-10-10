"""Read-only recovery subgraph: receipt, artifacts, ledger and concrete next steps."""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class RecoveryState(TypedDict, total=False):
    state_path: str
    business: dict
    receipt_report: dict
    checks: list[dict]
    report: dict


def build_execution_recovery_graph():
    def receipts(state):
        from phase_agent.tools.state.execution_receipts import recovery_report

        return {"receipt_report": recovery_report(state["state_path"], state["business"])}

    def artifacts(state):
        from phase_agent.tools.state.execution_recovery_evidence import inspect_execution

        return {
            "checks": [
                inspect_execution(state["business"], issue, state_path=state["state_path"])
                for issue in state["receipt_report"]["unsettled"]
            ]
        }

    def report(state):
        return {
            "report": {
                **state["receipt_report"],
                "checks": state["checks"],
                "automatic_replay_allowed": False,
            }
        }

    graph = StateGraph(RecoveryState)
    graph.add_node("read_execution_receipts", receipts)
    graph.add_node("inspect_registered_artifacts_and_ledger", artifacts)
    graph.add_node("propose_recovery_steps", report)
    graph.add_edge(START, "read_execution_receipts")
    graph.add_edge("read_execution_receipts", "inspect_registered_artifacts_and_ledger")
    graph.add_edge("inspect_registered_artifacts_and_ledger", "propose_recovery_steps")
    graph.add_edge("propose_recovery_steps", END)
    return graph.compile(checkpointer=False, name="execution_recovery")


def execution_recovery_report(state_path, state, *, graph=None):
    graph = graph or build_execution_recovery_graph()
    return graph.invoke({"state_path": str(state_path), "business": state})["report"]

"""Durable human approval only; this graph has no scientific side effects."""

import hashlib
import json
from pathlib import Path
from typing import TypedDict
from phase_agent.graphs.contracts import ApprovalIdentity


class ApprovalState(TypedDict, total=False):
    envelope: dict
    decision: dict
    delivered: bool


def build_approval_graph(checkpointer):
    from langgraph.graph import StateGraph, START, END
    from langgraph.types import interrupt

    def human_review(state):
        ApprovalIdentity.model_validate(state["envelope"])
        answer = interrupt(state["envelope"])
        if not isinstance(answer, dict) or answer.get("decision") not in {"approve", "reject"}:
            raise ValueError("invalid native approval decision")
        if answer.get("envelope") != state["envelope"]:
            raise ValueError("native approval identity mismatch")
        return {"decision": answer, "delivered": False}

    graph = StateGraph(ApprovalState)
    graph.add_node("human_review", human_review)
    graph.add_edge(START, "human_review")
    graph.add_edge("human_review", END)
    return graph.compile(checkpointer=checkpointer)


def durable_approval(state_path, envelope, decision=None):
    """Resume one immutable identity; consumed approvals cannot replay tools."""
    from filelock import FileLock
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.types import Command

    envelope = ApprovalIdentity.model_validate(envelope).model_dump()
    if decision not in {None, "approve", "reject"}:
        raise ValueError("invalid native approval decision")

    path = Path(state_path).resolve().parent / "langgraph_approvals.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = json.dumps(
        {"state_path": str(Path(state_path).resolve()), "envelope": envelope},
        sort_keys=True,
        ensure_ascii=False,
    )
    config = {"configurable": {"thread_id": hashlib.sha256(identity.encode()).hexdigest()}}
    with FileLock(str(path) + ".lock", timeout=10):
        with SqliteSaver.from_conn_string(str(path)) as saver:
            graph = build_approval_graph(saver)
            snapshot = graph.get_state(config)
            if not snapshot.values:
                graph.invoke({"envelope": envelope}, config)
                snapshot = graph.get_state(config)
            if snapshot.values.get("envelope") != envelope:
                raise ValueError("native approval identity mismatch")
            if decision is None:
                return {"status": "awaiting_approval", "checkpoint_path": str(path)}
            if snapshot.values.get("delivered"):
                return {"status": "approval_reconciliation_required", "checkpoint_path": str(path)}
            answer = {"decision": decision, "envelope": envelope}
            saved = snapshot.values.get("decision")
            if saved and saved != answer:
                raise ValueError("native approval decision mismatch")
            if not saved:
                graph.invoke(Command(resume=answer), config)
            graph.update_state(config, {"delivered": True}, as_node="human_review")
            return {"status": "decision_delivered", "checkpoint_path": str(path)}

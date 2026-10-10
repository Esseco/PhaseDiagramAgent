"""Durable scientific direction and immutable proposal approval."""

import hashlib
import json
from pathlib import Path
from typing import TypedDict


class DirectionState(TypedDict, total=False):
    proposal: dict
    decision: str
    direction: str
    status: str


def build_direction_graph(checkpointer):
    from langgraph.graph import StateGraph, START, END
    from langgraph.types import interrupt

    def review(state):
        answer = interrupt({"kind": "scientific_direction", "proposal": state["proposal"]})
        if not isinstance(answer, dict) or answer.get("proposal") != state["proposal"]:
            raise ValueError("direction_proposal_identity_mismatch")
        if answer.get("decision") not in {"approve", "reject"}:
            raise ValueError("invalid_direction_decision")
        return {"decision": answer["decision"]}

    def route(state):
        return "rejected" if state["decision"] == "reject" else state["direction"]

    def approved(state):
        return {"status": "approved"}

    graph = StateGraph(DirectionState)
    graph.add_node("review_direction_and_plan", review)
    for direction in ("activate", "supplement_dft", "other"):
        graph.add_node(direction, approved)
        graph.add_edge(direction, END)
    graph.add_node("rejected", lambda state: {"status": "rejected"})
    graph.add_edge("rejected", END)
    graph.add_edge(START, "review_direction_and_plan")
    graph.add_conditional_edges(
        "review_direction_and_plan", route, ["activate", "supplement_dft", "other", "rejected"]
    )
    return graph.compile(checkpointer=checkpointer, name="scientific_direction_review")


def direction_checkpoint(state_path, proposal, decision=None):
    from filelock import FileLock
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.types import Command

    direction = proposal["direction"]
    if direction not in {"activate", "supplement_dft", "other"}:
        raise ValueError("invalid_scientific_direction")
    if decision not in {None, "approve", "reject"}:
        raise ValueError("invalid_direction_decision")
    identity = hashlib.sha256(
        json.dumps(proposal, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    path = Path(state_path).resolve().parent / "langgraph_directions.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {"configurable": {"thread_id": identity}}
    with FileLock(str(path) + ".lock", timeout=10):
        with SqliteSaver.from_conn_string(str(path)) as saver:
            graph = build_direction_graph(saver)
            snapshot = graph.get_state(config)
            if not snapshot.values:
                graph.invoke({"proposal": proposal, "direction": direction}, config)
                snapshot = graph.get_state(config)
            if decision is not None:
                saved = snapshot.values.get("decision")
                if saved and saved != decision:
                    raise ValueError("direction_decision_already_recorded")
                if snapshot.next:
                    graph.invoke(
                        Command(resume={"proposal": proposal, "decision": decision}), config
                    )
            return dict(graph.get_state(config).values)

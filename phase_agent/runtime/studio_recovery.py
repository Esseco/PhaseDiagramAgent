"""Recover only a checkpointed failed read/analysis stage, never replay a tool."""

import json
import sqlite3
from copy import copy
from pathlib import Path

SAFE_NODES = {
    "collect_and_reconcile",
    "scientific_feedback",
    "edge_direction_review",
    "wait",
    "analyze",
}


def safe_read_checkpoint(snapshot):
    if not snapshot.next or any(task.interrupts for task in snapshot.tasks):
        return False
    if set(snapshot.next) <= SAFE_NODES:
        return True
    if tuple(snapshot.next) != ("observe",):
        return False
    children = [task.state for task in snapshot.tasks if hasattr(task.state, "next")]
    return bool(children) and all(safe_read_checkpoint(child) for child in children)


def can_resume_message(config_path, message_identity):
    if not config_path:
        return False
    config = Path(config_path)
    settings = json.loads(config.read_text(encoding="utf-8"))
    state_path = Path(settings["state_path"])
    if not state_path.is_absolute():
        state_path = config.parent / state_path
    path = state_path.resolve().parent / "langgraph_lifecycle.sqlite"
    if not path.is_file():
        return False
    from langgraph.checkpoint.sqlite import SqliteSaver
    from phase_agent.graphs.scientific_graph import scientific_graph

    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, check_same_thread=False)
    try:
        graph = copy(scientific_graph())
        graph.checkpointer = SqliteSaver(connection)
        snapshot = graph.get_state(
            {"configurable": {"thread_id": "project-scientific-lifecycle-v2"}}, subgraphs=True
        )
        return snapshot.values.get(
            "request_id"
        ) == "webui-" + message_identity and safe_read_checkpoint(snapshot)
    finally:
        connection.close()

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from phase_agent.graphs.project.runner import _rebind_pending_subgraphs, _validate_verified_resume


class State(TypedDict, total=False):
    current_node: str
    request_id: str


def test_rebind_first_child_node_preserves_its_pending_work():
    calls = []
    child = StateGraph(State)

    def collect(state):
        calls.append(state["request_id"])
        raise RuntimeError("simulated interrupted read")

    child.add_node("collect_and_reconcile", collect)
    child.add_edge(START, "collect_and_reconcile")
    child.add_edge("collect_and_reconcile", END)
    parent = StateGraph(State)
    parent.add_node("initialize", lambda state: {"current_node": "initialize_confirmed_run"})
    parent.add_node("observe", child.compile())
    parent.add_edge(START, "initialize")
    parent.add_edge("initialize", "observe")
    parent.add_edge("observe", END)
    graph = parent.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "synthetic"}}
    with pytest.raises(RuntimeError, match="interrupted read"):
        graph.invoke({"request_id": "old"}, config)
    before = graph.get_state(config, subgraphs=True)
    assert before.tasks[0].state.next == ("collect_and_reconcile",)
    assert before.tasks[0].state.values["current_node"] == "initialize_confirmed_run"
    _rebind_pending_subgraphs(graph, before, "new")
    after = graph.get_state(config, subgraphs=True)
    assert after.tasks[0].state.next == ("collect_and_reconcile",)
    assert after.tasks[0].state.values["request_id"] == "new"
    assert calls == ["old"]


def test_verified_resume_is_bound_to_exact_request_checkpoint_and_pending_node():
    import hashlib
    from types import SimpleNamespace

    blob = b"checkpoint contents"
    snapshot = SimpleNamespace(config={"configurable": {"checkpoint_id": "id"}},
                               values={"request_id": "request"}, next=("react",))
    proof = {"checkpoint_id": "id", "request_id": "request",
             "checkpoint_sha256": hashlib.sha256(blob).hexdigest()}
    _validate_verified_resume(snapshot, "request", proof, (blob,))
    for invalid in ({**proof, "request_id": "other"}, {**proof, "checkpoint_id": "other"},
                    {**proof, "checkpoint_sha256": "0" * 64}):
        with pytest.raises(ValueError, match="verified_resume_checkpoint_changed"):
            _validate_verified_resume(snapshot, "request", invalid, (blob,))
    snapshot.next = ("finalize",)
    with pytest.raises(ValueError, match="verified_resume_checkpoint_changed"):
        _validate_verified_resume(snapshot, "request", proof, (blob,))

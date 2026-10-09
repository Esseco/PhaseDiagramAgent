"""Exercise the production hierarchy with synthetic callbacks, never real data."""
import json
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from orchestration.scientific_graph import scientific_graph
from orchestration.search_workflow_graph import run_search_workflow_graph
from orchestration.event_loop_graph import run_event_loop_graph
from orchestration.tool_action_graph import run_tool_action_graph
from orchestration.studio_chat_graph import build_studio_graph


def test_production_hierarchy_is_recursively_discoverable():
    graph = build_studio_graph(lambda *_: "unused")
    children = dict(graph.get_subgraphs(recurse=True))
    assert "confirmed_local_chat" in children
    assert "confirmed_local_chat|bounded_action_graph" in children
    tools = children["confirmed_local_chat|bounded_action_graph|execute_validated_action"]
    assert children["confirmed_local_chat"].name == "scientific_lifecycle"
    assert children["confirmed_local_chat|bounded_action_graph"].name == "bounded_action_iteration"
    assert tools.name == "approved_scientific_action"
    assert {"approval_gate", "validate_action", "mlip_finetune", "mc_search", "dft_inputs"} <= set(tools.nodes)
    assert "tool__update_mlip" in dict(tools.get_subgraphs())["mlip_finetune"].nodes


def test_studio_runs_nested_science_and_keeps_full_results_out_of_stream(tmp_path):
    calls = []
    secret_runtime_object = object()

    def act(frame, *, event_graph):
        def execute(state, *, tool_graph):
            response = run_tool_action_graph(
                tool_graph=tool_graph, registry={"pause_search": {}},
                initialize=lambda: {"_graph_prepared": True},
                refresh=lambda value: value, prepare=lambda value: value,
                approve=lambda value: {"status": "awaiting_approval", "private": secret_runtime_object},
                validate=lambda value: calls.append("validate"),
                dispatch=lambda value: calls.append("dispatch"),
                finish=lambda value: calls.append("finish"))
            assert response["private"] is secret_runtime_object
            return {"response": response, "step_id": "synthetic-step"}

        def persist(state):
            assert state["response"]["private"] is secret_runtime_object
            calls.append("persist")
            return {"offset": state["offset"] + 1}

        run_event_loop_graph(max_steps=1, execute=execute, persist=persist,
                             route=lambda _: "__end__", event_graph=event_graph)
        return frame

    def sender(text, thread):
        result = run_search_workflow_graph(
            initialize=lambda: {"_workflow_prepared": True}, collect=lambda frame: frame,
            analyze=lambda frame: frame, wait=lambda frame: frame,
            assess=lambda frame: frame, act=act,
            finalize=lambda frame: {"status": "awaiting_approval", "private": secret_runtime_object})
        assert result["private"] is secret_runtime_object
        return "awaiting approval"

    graph = build_studio_graph(sender, receipt_path=tmp_path / "gateway.json", checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "synthetic-thread"}}
    updates = list(graph.stream({"messages": [HumanMessage(content="继续", id="m")]}, config,
                                stream_mode="updates", subgraphs=True))
    json.dumps(updates, default=lambda value: value.model_dump())
    assert calls == ["persist"]  # Waiting for approval must not reach tools.
    names = {name for _, update in updates for name in update}
    assert {"collect_and_reconcile", "scientific_feedback", "approval_gate",
            "persist_action_outcome"} <= names
    assert all("private" not in str(update) for _, update in updates)
    assert scientific_graph() is scientific_graph()

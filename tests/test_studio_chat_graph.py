import pytest
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from phase_agent.graphs.studio_chat_graph import build_studio_graph


def test_studio_direct_message_and_history_replay(tmp_path):
    calls = []
    graph = build_studio_graph(
        lambda text, thread: calls.append(text) or "ok",
        receipt_path=tmp_path / "gateway.json",
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "t"}}
    initial = graph.invoke({"messages": [HumanMessage(content="继续", id="m")]}, config)
    assert "__interrupt__" not in initial
    assert initial["messages"][-1].content == "ok"
    assert calls == ["继续"]
    graph = build_studio_graph(
        lambda text, thread: calls.append(text) or "ok",
        receipt_path=tmp_path / "gateway.json",
        checkpointer=InMemorySaver(),
    )
    replay = graph.invoke({"messages": [HumanMessage(content="继续", id="m")]}, config)
    assert "禁止历史重放" in replay["messages"][-1].content
    assert calls == ["继续"]


def test_new_message_same_text_is_a_new_turn(tmp_path):
    calls = []
    graph = build_studio_graph(
        lambda text, thread: calls.append(text) or "ok",
        receipt_path=tmp_path / "gateway.json",
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "t"}}
    for identity in ("m1", "m2"):
        result = graph.invoke({"messages": [HumanMessage(content="继续", id=identity)]}, config)
        assert "__interrupt__" not in result
    assert calls == ["继续", "继续"]


def test_studio_text_blocks_sent_as_text(tmp_path):
    calls = []
    graph = build_studio_graph(
        lambda text, thread: calls.append(text) or "ok",
        receipt_path=tmp_path / "gateway.json",
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "blocks"}}
    result = graph.invoke(
        {"messages": [{"role": "user", "id": "m", "content": [{"type": "text", "text": "继续"}]}]},
        config,
    )
    assert "__interrupt__" not in result
    assert calls == ["继续"]


def test_failed_send_is_not_retried_automatically(tmp_path):
    calls = []

    def fail(text, thread):
        calls.append(text)
        raise RuntimeError("synthetic failure")

    graph = build_studio_graph(fail, receipt_path=tmp_path / "gateway.json")
    config = {"configurable": {"thread_id": "t"}}
    message = {"messages": [HumanMessage(content="同意", id="m")]}
    with pytest.raises(RuntimeError, match="synthetic failure"):
        graph.invoke(message, config)
    result = graph.invoke(message, config)
    assert "禁止历史重放" in result["messages"][-1].content
    assert calls == ["同意"]


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [HumanMessage(content="  ")],
        [HumanMessage(content="旧指令"), AIMessage(content="旧回复")],
        [HumanMessage(content=[{"type": "image_url", "image_url": {"url": "test"}}])],
    ],
)
def test_invalid_or_old_turn_is_not_dispatched(messages, tmp_path):
    calls = []
    graph = build_studio_graph(
        lambda *args: calls.append(args), receipt_path=tmp_path / "gateway.json"
    )
    result = graph.invoke({"messages": messages}, {"configurable": {"thread_id": "invalid"}})
    assert result["messages"][-1].content.startswith("未执行：")
    assert not calls
    assert "__interrupt__" not in result


def test_chat_output_only_contains_messages(tmp_path):
    graph = build_studio_graph(
        lambda text, thread: "current status", receipt_path=tmp_path / "gateway.json"
    )
    result = graph.invoke(
        {"messages": [HumanMessage(content="status", id="brief")]},
        {"configurable": {"thread_id": "brief"}},
    )
    assert set(result) == {"messages"}
    assert result["messages"][-1].content == "current status"


def test_chat_input_form_only_requests_messages(tmp_path):
    graph = build_studio_graph(lambda *args: "ok", receipt_path=tmp_path / "gateway.json")
    schema = graph.get_input_jsonschema()
    assert set(schema["properties"]) == {"messages"}


def test_chat_stream_does_not_emit_status_dictionary(tmp_path):
    graph = build_studio_graph(lambda *args: "简洁答复", receipt_path=tmp_path / "gateway.json")
    updates = list(
        graph.stream(
            {"messages": [HumanMessage(content="状态", id="stream")]},
            {"configurable": {"thread_id": "stream"}},
            stream_mode="updates",
        )
    )
    assert [next(iter(update)) for update in updates] == ["receive_message", "agent", "respond"]
    assert updates[-1]["respond"] in (None, {})
    assert set(updates[-2]["agent"]) == {"messages", "reply"}
    assert {"receive_message", "agent", "respond"} <= set(graph.get_graph().nodes)
    assert "confirmed_local_chat" not in graph.get_graph().nodes
    assert "read_persisted_project_status" not in graph.get_graph().nodes


def test_real_dialogue_nodes_are_visible_and_answer_skips_science(tmp_path):
    from tests.test_unified_react_dialogue import make_handler
    from phase_agent.graphs.dialogue.graph import build_project_turn_graph
    from phase_agent.graphs.scientific_graph import scientific_graph
    from phase_agent.tools.step_runner.file_protocol import read_json

    calls = []

    def model(payload):
        calls.append(payload)
        return {"kind": "answer", "answer": "这是状态说明，不执行计算。"}

    handler = make_handler(tmp_path, model)
    graph = build_studio_graph(
        lambda text, thread: handler([{"role": "user", "content": text}], conversation_id=thread),
        receipt_path=tmp_path / "gateway.json",
    )
    assert dict(graph.get_subgraphs())["agent"] is build_project_turn_graph(scientific_graph())
    updates = list(
        graph.stream(
            {"messages": [HumanMessage(content="解释当前状态", id="question")]},
            {"configurable": {"thread_id": "dialogue"}},
            stream_mode="updates",
            subgraphs=True,
        )
    )
    names = {name for _, update in updates for name in update}
    assert {"receive_message", "decide", "prepare_context", "present", "respond"} <= names
    assert "scientific_workflow" not in names
    assert len(calls) == 1
    assert not read_json(handler.state_path, {}).get("pending_execution_policies")


def test_checkpoint_resume_after_failed_send_does_not_reexecute(tmp_path):
    calls = []

    def fail(text, thread):
        calls.append(text)
        raise RuntimeError("failed sender")

    graph = build_studio_graph(
        fail, receipt_path=tmp_path / "gateway.json", checkpointer=InMemorySaver()
    )
    config = {"configurable": {"thread_id": "failed-resume"}}
    with pytest.raises(RuntimeError, match="failed sender"):
        graph.invoke({"messages": [HumanMessage(content="同意", id="approval")]}, config)
    result = graph.invoke(None, config)
    assert "禁止历史重放" in result["messages"][-1].content
    assert calls == ["同意"]

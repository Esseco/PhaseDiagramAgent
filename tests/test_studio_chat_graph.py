import pytest
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from orchestration.studio_chat_graph import build_studio_graph


def test_studio_direct_message_and_history_replay(tmp_path):
    calls = []
    graph = build_studio_graph(lambda text, thread: calls.append(text) or "ok",
                              receipt_path=tmp_path / "gateway.json", checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t"}}
    initial = graph.invoke({"messages": [HumanMessage(content="继续", id="m")]}, config)
    assert "__interrupt__" not in initial
    assert initial["messages"][-1].content == "ok"
    assert calls == ["继续"]
    graph = build_studio_graph(lambda text, thread: calls.append(text) or "ok",
                              receipt_path=tmp_path / "gateway.json", checkpointer=InMemorySaver())
    replay = graph.invoke({"messages": [HumanMessage(content="继续", id="m")]}, config)
    assert "禁止历史重放" in replay["messages"][-1].content
    assert calls == ["继续"]


def test_new_message_same_text_is_a_new_turn(tmp_path):
    calls = []
    graph = build_studio_graph(lambda text, thread: calls.append(text) or "ok", receipt_path=tmp_path / "gateway.json",
                              checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t"}}
    for identity in ("m1", "m2"):
        result = graph.invoke({"messages": [HumanMessage(content="继续", id=identity)]}, config)
        assert "__interrupt__" not in result
    assert calls == ["继续", "继续"]


def test_studio_text_blocks_sent_as_text(tmp_path):
    calls = []
    graph = build_studio_graph(lambda text, thread: calls.append(text) or "ok",
        receipt_path=tmp_path / "gateway.json", checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "blocks"}}
    result = graph.invoke({"messages": [{"role": "user", "id": "m",
        "content": [{"type": "text", "text": "继续"}]}]}, config)
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


@pytest.mark.parametrize("messages", [[], [HumanMessage(content="  ")],
    [HumanMessage(content="旧指令"), AIMessage(content="旧回复")],
    [HumanMessage(content=[{"type": "image_url", "image_url": {"url": "test"}}])]])
def test_invalid_or_old_turn_is_not_dispatched(messages, tmp_path):
    calls = []
    graph = build_studio_graph(lambda *args: calls.append(args), receipt_path=tmp_path / "gateway.json")
    result = graph.invoke({"messages": messages}, {"configurable": {"thread_id": "invalid"}})
    assert result["messages"][-1].content.startswith("未执行：")
    assert not calls
    assert "__interrupt__" not in result

import asyncio
from copy import deepcopy
from contextlib import asynccontextmanager
import json
import os
from threading import Event
from uuid import uuid4

os.environ.setdefault("REDIS_URI", "fake")
os.environ.setdefault("LANGGRAPH_RUNTIME_EDITION", "inmem")

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from phase_agent.graphs.studio_chat_graph import build_studio_graph
from phase_agent.graphs.cancellation import cancellation_scope, ScientificRunCancelled
from phase_agent.runtime.studio_payloads import guarded_studio_stream, transport_view


@pytest.mark.parametrize("cancel_after_return", [False, True])
@pytest.mark.parametrize("rollback_ui_checkpoint", [False, True])
def test_lost_completed_reply_is_redelivered_without_running_action(
    tmp_path, cancel_after_return, rollback_ui_checkpoint
):
    calls = []
    signal = Event()

    def sender(text, thread):
        calls.append(text)
        (tmp_path / "prepared.txt").write_text("completed")
        if cancel_after_return:
            signal.set()
        return "刷新输入已生成，等待结果"

    saver = InMemorySaver()
    config = {"configurable": {"thread_id": "delivery"}}
    message = {"messages": [HumanMessage(content="同意", id="approval")]}
    graph = build_studio_graph(sender, receipt_path=tmp_path / "gateway.json", checkpointer=saver)
    if cancel_after_return:
        with cancellation_scope(signal), pytest.raises(ScientificRunCancelled):
            graph.invoke(message, config)
    else:
        first = graph.invoke(message, config)
    # Studio uses rollback for a repeated run. Its UI checkpoint may disappear,
    # while the independently committed gateway receipt must retain the answer.
    graph = build_studio_graph(
        lambda *args: pytest.fail("completed action must never run twice"),
        receipt_path=tmp_path / "gateway.json",
        checkpointer=InMemorySaver() if rollback_ui_checkpoint else saver,
    )
    recovered = graph.invoke(message, config)
    assert recovered["messages"][-1].content == "刷新输入已生成，等待结果"
    assert calls == ["同意"]
    if not cancel_after_return:
        assert recovered["messages"][-1].id == first["messages"][-1].id
        assert len([m for m in recovered["messages"] if m.type == "ai"]) == 1


def test_changed_text_with_same_id_cannot_recover_another_reply(tmp_path):
    graph = build_studio_graph(
        lambda *args: "old completed", receipt_path=tmp_path / "gateway.json"
    )
    config = {"configurable": {"thread_id": "different-text"}}
    graph.invoke({"messages": [HumanMessage(content="同意", id="same")]}, config)
    replay = graph.invoke({"messages": [HumanMessage(content="生成别的", id="same")]}, config)
    assert "禁止历史重放" in replay["messages"][-1].content


@pytest.mark.parametrize("key", ["state", "loaded_state", "facts"])
def test_transport_removes_full_scientific_snapshots_from_child_results(key):
    state = {
        "tasks": [{"task_id": "t", "status": "completed"}],
        "active_model_version": "new",
        "fixture_blob": "x" * 200000,
    }
    event = {
        "type": "task_result",
        "payload": {"result": [("result", {key: state, "reply": "fresh answer"})]},
    }
    before = deepcopy(event)
    projected = transport_view(event)
    assert event == before
    assert len(json.dumps(projected)) < 2000
    assert "fresh answer" in json.dumps(projected)


@pytest.mark.parametrize(
    "requested", [["debug", "messages"], ["messages"], ["debug", "messages-tuple"]]
)
def test_real_api_emits_latest_terminal_state_after_approval(tmp_path, monkeypatch, requested):
    from langgraph_api import stream
    from langgraph_api.asyncio import ValueEvent
    from tests.test_unified_react_dialogue import make_handler

    calls = []
    handler = make_handler(
        tmp_path,
        lambda _: {
            "tool": "check_convergence",
            "task_key": "inspect",
            "parameters": {},
            "reason": "核对已有数据",
            "budget": 0,
        },
        {
            "check_convergence": lambda **_: (
                calls.append("executed") or {"converged": False, "budget_exhausted": False}
            )
        },
    )
    # Preserve the actual chat, approval, lifecycle and installed API encoder.
    graph = build_studio_graph(
        lambda text, thread: handler([{"role": "user", "content": text}], conversation_id=thread),
        receipt_path=tmp_path / "gateway.json",
        checkpointer=InMemorySaver(),
    )

    @asynccontextmanager
    async def get_graph(*args, **kwargs):
        yield graph

    async def get_store():
        return None

    async def get_checkpointer():
        return InMemorySaver()

    monkeypatch.setattr(stream, "get_graph", get_graph)
    monkeypatch.setattr(stream.api_store, "get_store", get_store)
    monkeypatch.setattr(stream.api_checkpointer, "get_checkpointer", get_checkpointer)

    async def run():
        for index, text in enumerate(["分析下一步", "同意", "同意"]):
            request = {
                "run_id": uuid4(),
                "kwargs": {
                    "input": {"messages": [HumanMessage(content=text, id=f"u{min(index, 1)}")]},
                    "command": None,
                    "config": {
                        "configurable": {"graph_id": "phase_chat", "thread_id": "wire"},
                        "metadata": {},
                    },
                    "stream_mode": requested,
                    "subgraphs": True,
                    "temporary": False,
                    "durability": "async",
                    "checkpoint_during": True,
                },
            }
            events = [
                row
                async for row in guarded_studio_stream(
                    stream.astream_state, request, 1, ValueEvent()
                )
            ]
            assert events[-1][0] == "values", [name for name, _ in events[-4:]]
            final = events[-1][1]["messages"][-1]
            saved = (await graph.aget_state({"configurable": {"thread_id": "wire"}})).values[
                "messages"
            ][-1]
            assert final.id == saved.id and final.content == saved.content
            if index == 0:
                assert "同意" in final.content and calls == []
            else:
                assert "已完成" in final.content and "待确认" not in final.content
                assert calls == ["executed"]

    asyncio.run(run())


def test_failed_stream_never_emits_a_successful_terminal_frame():
    request = {
        "kwargs": {
            "input": {"messages": ["approve"]},
            "stream_mode": ["debug", "messages"],
            "config": {"configurable": {"graph_id": "phase_chat"}},
        }
    }

    async def broken(run, **kwargs):
        yield "debug", {"type": "task", "payload": {}}
        raise RuntimeError("stream failed")

    async def run():
        events = []
        with pytest.raises(RuntimeError, match="stream failed"):
            async for event in guarded_studio_stream(broken, request):
                events.append(event)
        assert not any(name == "values" for name, _ in events)

    asyncio.run(run())


def test_graph_task_state_and_interrupts_are_not_scientific_snapshot():
    checkpoint = {
        "values": {"messages": ["current"]},
        "tasks": [{"interrupts": ["review"]}],
        "next": ["review"],
    }
    event = {"state": checkpoint}
    assert transport_view(event) is event


@pytest.mark.parametrize(
    "variant", ["v2", "other_graph", "already_values", "interrupted", "failed", "child_last"]
)
def test_terminal_frame_respects_protocol_and_root_boundaries(variant):
    request = {
        "kwargs": {
            "input": {"messages": ["approve"]},
            "stream_mode": ["debug", "messages"],
            "config": {"configurable": {"graph_id": "phase_chat"}},
        }
    }
    if variant == "v2":
        request["kwargs"]["config"]["configurable"]["__event_streaming_v2"] = True
    if variant == "other_graph":
        request["kwargs"]["config"]["configurable"]["graph_id"] = "phase_status"
    if variant == "already_values":
        request["kwargs"]["stream_mode"].append("values")
    root = {
        "config": {"configurable": {"checkpoint_ns": ""}},
        "values": {"messages": ["latest root"]},
        "next": [],
        "tasks": [],
    }
    if variant in {"interrupted", "failed"}:
        root["tasks"] = (
            [{"interrupts": ["review"]}] if variant == "interrupted" else [{"error": "failed"}]
        )

    async def original(prepared, **kwargs):
        kwargs.get("on_checkpoint", lambda _: None)(root)
        if variant == "child_last":
            kwargs["on_checkpoint"](
                {
                    "config": {"configurable": {"checkpoint_ns": "agent:child"}},
                    "values": {"messages": ["stale child"]},
                    "next": [],
                }
            )
        yield "debug", {"type": "checkpoint", "payload": root}

    async def collect():
        return [row async for row in guarded_studio_stream(original, request)]

    events = asyncio.run(collect())
    if variant == "child_last":
        assert events[-1] == ("values", {"messages": ["latest root"]})
    else:
        assert not any(name == "values" for name, _ in events)

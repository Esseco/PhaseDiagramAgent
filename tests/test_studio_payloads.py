import asyncio
import copy
import os

os.environ.setdefault("REDIS_URI", "fake")
os.environ.setdefault("LANGGRAPH_RUNTIME_EDITION", "inmem")

from phase_agent.runtime import studio_payloads as module


def test_projection_preserves_business_data_and_messages():
    original = {
        "values": {
            "facts": {
                "tasks": [{"id": "T1", "status": "running"}],
                "event_history": ["x" * 1000000],
            },
            "messages": [{"content": "answer"}],
        },
        "interrupts": {"task": ["approve"]},
        "status": "error",
    }
    before = copy.deepcopy(original)
    projected = module.transport_view(original)
    assert original == before
    assert len(str(projected)) < 1000
    assert projected["values"]["messages"] is original["values"]["messages"]
    assert projected["interrupts"] == original["interrupts"]
    assert module.transport_view({"messages": ["ok"]}) == {"messages": ["ok"]}


def test_root_checkpoint_namespace():
    assert module.root_checkpoint({"config": {"configurable": {"checkpoint_ns": ""}}})
    assert not module.root_checkpoint(
        {"config": {"configurable": {"checkpoint_ns": "agent:child"}}}
    )
    assert not module.root_checkpoint({"checkpoint": {"checkpoint_ns": "agent:child"}})


def test_agent_reply_update_is_hidden_when_respond_emits_final_message():
    assert module.is_agent_reply_update({"agent": {"reply": "same"}})
    assert not module.is_agent_reply_update({"agent": {"diagnostic": "keep"}})
    assert not module.is_agent_reply_update({"respond": {"messages": ["same"]}})

    from phase_agent.runtime.studio_payloads import guarded_studio_stream

    run = {
        "kwargs": {
            "input": None,
            "command": None,
            "stream_mode": ["updates", "messages"],
            "config": {"configurable": {"graph_id": "phase_chat"}},
        }
    }

    async def original(prepared, *args, **kwargs):
        yield "updates", {"agent": {"reply": "same"}}
        yield "updates", {"respond": {"messages": ["same"]}}
        yield "messages", {"reply": "same"}

    async def collect():
        return [row async for row in guarded_studio_stream(original, run)]

    assert asyncio.run(collect()) == [
        ("updates", {"respond": {"messages": ["same"]}}),
        ("messages", {"reply": "same"}),
    ]


def test_phase_chat_transport_deduplicates_agent_and_respond_reply():
    run = {
        "kwargs": {
            "input": {"messages": ["hello"]},
            "stream_mode": ["updates"],
            "config": {"configurable": {"graph_id": "phase_chat"}},
        }
    }

    async def original(prepared, *args, **kwargs):
        yield "updates", {"agent": {"reply": "same answer"}}
        yield "updates", {"respond": {"messages": ["same answer"]}}

    async def collect():
        return [row async for row in module.guarded_studio_stream(original, run)]

    assert asyncio.run(collect()) == [("updates", {"respond": {"messages": ["same answer"]}})]


def test_adapter_filters_child_cache_updates_and_projects_legacy_search(monkeypatch):
    from langgraph_api import stream
    from langgraph_api import worker_runtime
    from langgraph_runtime_inmem.ops import Threads

    root = {
        "config": {"configurable": {"checkpoint_ns": ""}},
        "values": {"messages": ["ok"]},
    }
    child = {
        "config": {"configurable": {"checkpoint_ns": "agent:child"}},
        "values": {"facts": {"tasks": ["original"]}},
    }

    closed = []

    async def fake_stream(*args, on_checkpoint, **kwargs):
        on_checkpoint(root)
        on_checkpoint(child)
        try:
            yield "debug|agent:child", child
        finally:
            closed.append(True)

    async def fake_search(*args, **kwargs):
        async def rows():
            yield {"values": child["values"], "status": "error"}

        return rows(), 42

    async def fake_get(*args, **kwargs):
        rows, _ = await fake_search()
        return rows

    monkeypatch.setattr(stream, "astream_state", fake_stream)
    monkeypatch.setattr(worker_runtime, "astream_state", fake_stream)
    monkeypatch.setattr(Threads, "search", fake_search)
    monkeypatch.setattr(Threads, "get", fake_get)
    module.install_studio_payload_views()
    installed = stream.astream_state
    module.install_studio_payload_views()
    assert stream.astream_state is installed

    async def run():
        observed = []
        events = [
            event
            async for _, event in stream.astream_state(on_checkpoint=observed.append)
        ]
        assert observed == [root]
        assert events[0]["values"]["facts"]["field_count"] == 1
        rows, cursor = await Threads.search()
        assert cursor == 42
        assert [row async for row in rows][0]["status"] == "error"
        single = [row async for row in await Threads.get()]
        assert single[0]["values"]["facts"]["field_count"] == 1
        assert child["values"]["facts"] == {"tasks": ["original"]}
        assert closed == [True]
        pending = stream.astream_state()
        await anext(pending)
        await pending.aclose()
        assert closed == [True, True]

    asyncio.run(run())

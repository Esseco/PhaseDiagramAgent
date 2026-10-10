import asyncio
import threading
import time
from langchain_core.messages import HumanMessage
from phase_agent.graphs.studio_chat_graph import build_studio_graph
from phase_agent.graphs.cancellation import check_cancelled, cancellable_worker


def test_async_run_cancel_waits_for_worker_and_blocks_next_action(tmp_path):
    entered = threading.Event()
    stopped = threading.Event()
    actions = []
    def sender(text, thread):
        if text == "slow":
            entered.set()
            try:
                while True:
                    time.sleep(.01)
                    check_cancelled()
            finally:
                stopped.set()
        actions.append(text)
        return "done"
    graph = build_studio_graph(sender, receipt_path=tmp_path / "gateway.json")
    config = {"configurable": {"thread_id": "cancel"}}
    async def run():
        task = asyncio.create_task(graph.ainvoke({"messages": [HumanMessage(content="slow", id="m1")]}, config))
        while not entered.is_set():
            await asyncio.sleep(.01)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert stopped.is_set()
        assert actions == []
        result = await graph.ainvoke({"messages": [HumanMessage(content="next", id="m2")]}, config)
        assert result["messages"][-1].content == "done"
        assert actions == ["next"]
    asyncio.run(run())


def test_cancel_does_not_release_before_blocking_call_returns():
    entered = threading.Event()
    exited = threading.Event()
    def work():
        entered.set()
        time.sleep(.1)
        exited.set()
        check_cancelled()
    async def run():
        task = asyncio.create_task(cancellable_worker(work))
        while not entered.is_set():
            await asyncio.sleep(.005)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert exited.is_set()
    asyncio.run(run())


def test_cancel_before_dispatch_does_not_create_execution_receipt(tmp_path):
    import pytest
    from threading import Event
    from phase_agent.graphs.cancellation import cancellation_scope, ScientificRunCancelled
    from phase_agent.tools.dispatch.execute_tool_action import execute_tool_action
    signal = Event()
    signal.set()
    calls = []
    with cancellation_scope(signal), pytest.raises(ScientificRunCancelled):
        execute_tool_action({"tool": "adjust_strategy"}, registry={"adjust_strategy": {
            "handler": lambda **kwargs: calls.append("executed")}},
            context={"state_path": str(tmp_path / "state.json")})
    assert calls == []
    assert not list(tmp_path.iterdir())

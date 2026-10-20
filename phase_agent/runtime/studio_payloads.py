"""Keep scientific facts in checkpoints, out of Studio transport/cache views."""

from functools import wraps
from contextlib import aclosing


def transport_view(value):
    """Copy only changed containers; never edit saved scientific state."""
    if isinstance(value, dict):
        result = {}
        changed = False
        for key, item in value.items():
            if key == "facts" and isinstance(item, dict):
                projected = {
                    "_studio_view": "Full scientific facts retained in project state and checkpoints",
                    "field_count": len(item),
                }
            else:
                projected = transport_view(item)
            changed |= projected is not item
            result[key] = projected
        return result if changed else value
    if isinstance(value, (list, tuple)):
        items = [transport_view(item) for item in value]
        if any(a is not b for a, b in zip(items, value)):
            return tuple(items) if isinstance(value, tuple) else items
    return value


def root_checkpoint(checkpoint):
    if checkpoint is None:
        return True
    config = checkpoint.get("config", {}).get("configurable", {})
    namespace = config.get("checkpoint_ns")
    if namespace is None:
        namespace = (checkpoint.get("checkpoint") or {}).get("checkpoint_ns", "")
    return not namespace


def thread_view(row):
    values = row.get("values")
    projected = transport_view(values)
    return {**row, "values": projected} if projected is not values else row


async def guarded_studio_stream(original, run=None, *args, **kwargs):
    """Do not present a completed, empty Chat resume as a processed new message."""
    options = (run or {}).get("kwargs") or {}
    graph = ((options.get("config") or {}).get("configurable") or {}).get("graph_id")
    guard = (
        graph == "phase_chat" and options.get("input") is None and options.get("command") is None
    )
    requested = options.get("stream_mode") or []
    prepared = run
    if guard:
        prepared = {
            **run,
            "kwargs": {**options, "stream_mode": list(dict.fromkeys([*requested, "updates"]))},
        }
    executed = False
    stream = original(prepared, *args, **kwargs) if run is not None else original(*args, **kwargs)
    async with aclosing(stream) as events:
        async for name, event in events:
            if guard and name.split("|", 1)[0] == "updates":
                executed = True
                if "updates" not in requested:
                    continue
            yield name, event
    if guard and not executed:
        raise ValueError(
            "没有可恢复节点，本次未处理新消息。请在 Studio 的 Input/messages 中发送新的用户消息；仅修改历史消息后空续跑不会触发结果回收。"
        )


def install_studio_payload_views():
    """Adapt the installed local dev API without changing its package files."""
    from langgraph_api import stream
    from langgraph_api import worker_runtime
    from langgraph_runtime_inmem.ops import Threads

    original = stream.astream_state
    if getattr(original, "_phase_payload_view", False):
        return

    @wraps(original)
    async def astream_state(*args, on_checkpoint=lambda _: None, **kwargs):
        def observe(checkpoint):
            # A child debug event is not the root thread's latest state.
            if root_checkpoint(checkpoint):
                on_checkpoint(checkpoint)

        async with aclosing(
            guarded_studio_stream(original, *args, on_checkpoint=observe, **kwargs)
        ) as events:
            async for name, event in events:
                yield name, transport_view(event)

    astream_state._phase_payload_view = True
    stream.astream_state = astream_state
    worker_runtime.astream_state = astream_state
    search = Threads.search
    get = Threads.get

    @wraps(search)
    async def search_view(*args, **kwargs):
        rows, cursor = await search(*args, **kwargs)

        async def projected():
            async for row in rows:
                yield thread_view(row)

        return projected(), cursor

    Threads.search = staticmethod(search_view)

    @wraps(get)
    async def get_view(*args, **kwargs):
        rows = await get(*args, **kwargs)

        async def projected():
            async for row in rows:
                yield thread_view(row)

        return projected()

    Threads.get = staticmethod(get_view)

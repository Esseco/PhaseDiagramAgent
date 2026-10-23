"""Native Studio deletion with real lazy SQLite, without services or project data."""

import asyncio
import sys
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

from blockbuster import blockbuster_ctx
from langgraph.checkpoint.memory import InMemorySaver
import pytest

from phase_agent.persistence.studio_checkpoint_store import checkpoint_container
from phase_agent.persistence.studio_storage import install_memory_adapter, install_mutation_adapters
from tests.test_studio_storage import checkpoint


@pytest.fixture
def native(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from langgraph_runtime_inmem import checkpoint as runtime, ops
    import langgraph_api

    config = SimpleNamespace(USE_CUSTOM_CHECKPOINTER=False)
    monkeypatch.setitem(sys.modules, "langgraph_api.config", config)
    monkeypatch.setattr(langgraph_api, "config", config, raising=False)
    monkeypatch.setattr(runtime, "PersistentDict", checkpoint_container)
    factory = SimpleNamespace(InMemorySaver=runtime.InMemorySaver, MEMORY=None)
    install_memory_adapter(factory)
    memory = factory.InMemorySaver(serde=InMemorySaver().serde)
    factory.MEMORY = memory
    monkeypatch.setattr(ops, "Checkpointer", lambda **kw: memory)
    monkeypatch.setattr(ops.Threads, "handle_event", AsyncMock(return_value=None))
    # Remember the native globals before installation so every test restores them.
    monkeypatch.setattr(ops.Threads, "copy", ops.Threads.copy)
    monkeypatch.setattr(ops.Threads, "delete", ops.Threads.delete)
    monkeypatch.setattr(ops, "_delete_checkpoints_for_thread", ops._delete_checkpoints_for_thread)
    install_mutation_adapters(factory, ops)
    ids = [UUID(int=n) for n in (1, 2, 3)]
    conn = SimpleNamespace(
        store={
            "threads": [
                {"thread_id": tid, "metadata": {"graph_id": "phase_chat"}, "status": "idle"}
                for tid in ids
            ],
            "runs": [
                {"thread_id": tid, "run_id": UUID(int=10 + i), "status": "success"}
                for i, tid in enumerate(ids)
            ],
            "crons": [{"thread_id": str(tid)} for tid in ids],
        }
    )
    saved = {}
    for i, tid in enumerate(ids):
        saved[tid] = memory.put(
            {"configurable": {"thread_id": str(tid), "checkpoint_ns": ""}},
            checkpoint(i, {"dialogue": str(tid)}),
            {"run_id": "keep"},
            {"business_frame": i},
        )
    memory.put_writes(saved[ids[-1]], [("__interrupt__", {"approval": "retain"})], "human")
    try:
        yield SimpleNamespace(
            memory=memory, ops=ops, conn=conn, ids=ids, saved=saved, factory=factory, config=config
        )
    finally:
        for store in (memory.storage, memory.writes, memory.blobs):
            store.backend.conn.close()


def test_native_delete_works_under_blocking_detector_and_preserves_other_threads(native, tmp_path):
    n = native
    stores = (n.memory.storage, n.memory.writes, n.memory.blobs)
    other = n.memory.get_tuple(n.saved[n.ids[-1]])
    business = tmp_path / "state.json"
    business.write_text('{"tasks":[{"status":"running"}],"pending_execution_policies":{"keep":{}}}')
    before = business.read_bytes()

    async def run():
        with blockbuster_ctx(scanned_modules=["phase_agent", "langgraph_runtime_inmem"]):
            result = await n.ops.Threads.delete(n.conn, n.ids[0])
            assert await anext(result) == n.ids[0]

    asyncio.run(run())
    assert all(
        before is after
        for before, after in zip(stores, (n.memory.storage, n.memory.writes, n.memory.blobs))
    )
    assert str(n.ids[0]) not in n.memory.storage
    assert not any(k[0] == str(n.ids[0]) for k in n.memory.writes)
    assert not any(k[0] == str(n.ids[0]) for k in n.memory.blobs)
    assert n.memory.get_tuple(n.saved[n.ids[-1]]) == other
    assert {row["thread_id"] for row in n.conn.store["threads"]} == set(n.ids[1:])
    assert {row["thread_id"] for row in n.conn.store["runs"]} == set(n.ids[1:])
    assert business.read_bytes() == before
    assert n.memory._phase_storage.readers == 0


def test_run_scoped_delete_preserves_other_run_checkpoints(native):
    n = native
    tid = str(n.ids[0])
    n.memory.put(
        n.saved[n.ids[0]], checkpoint(9, "remove"), {"run_id": "remove"}, {"business_frame": 9}
    )

    async def run():
        with blockbuster_ctx(scanned_modules=["phase_agent", "langgraph_runtime_inmem"]):
            await n.ops._delete_checkpoints_for_thread(tid, n.conn, run_id="remove")

    asyncio.run(run())
    assert set(n.memory.storage[tid][""]) == {"0000"}
    assert n.memory.get_tuple(n.saved[n.ids[-1]]).pending_writes[0][2] == {"approval": "retain"}


def test_parallel_native_deletes_do_not_remove_wrong_metadata(native, monkeypatch):
    n = native
    original = n.ops._delete_checkpoints_for_thread

    async def yielding(*args, **kwargs):
        await asyncio.sleep(0)
        return await original(*args, **kwargs)

    monkeypatch.setattr(n.ops, "_delete_checkpoints_for_thread", yielding)

    async def run():
        results = await asyncio.gather(*(n.ops.Threads.delete(n.conn, tid) for tid in n.ids[:2]))
        return [await anext(row) for row in results]

    assert set(asyncio.run(run())) == set(n.ids[:2])
    assert [row["thread_id"] for row in n.conn.store["threads"]] == n.ids[2:]
    assert n.memory.get_tuple(n.saved[n.ids[-1]]).pending_writes[0][2] == {"approval": "retain"}


def test_thread_without_checkpoint_can_be_deleted_without_creating_a_namespace(native):
    n = native
    tid = UUID(int=99)
    n.conn.store["threads"].append({"thread_id": tid, "metadata": {}, "status": "idle"})

    async def run():
        with blockbuster_ctx(scanned_modules=["phase_agent", "langgraph_runtime_inmem"]):
            rows = await n.ops.Threads.delete(n.conn, tid)
            assert await anext(rows) == tid

    asyncio.run(run())
    assert str(tid) not in n.memory.storage
    assert len(n.conn.store["threads"]) == 3


def test_unknown_thread_keeps_all_stored_metadata(native):
    from starlette.exceptions import HTTPException

    n = native
    before = deepcopy(n.conn.store)
    with pytest.raises(HTTPException) as error:
        asyncio.run(n.ops.Threads.delete(n.conn, UUID(int=99)))
    assert error.value.status_code == 404
    assert n.conn.store == before


def test_first_delete_loads_checkpointer_outside_event_loop(native, monkeypatch):
    import threading

    n = native
    n.factory.MEMORY = None
    caller = threading.get_ident()

    def factory():
        assert threading.get_ident() != caller
        assert len(n.memory.storage) == 3
        n.factory.MEMORY = n.memory
        return n.memory

    monkeypatch.setattr(n.ops, "Checkpointer", factory)

    async def run():
        with blockbuster_ctx(scanned_modules=["phase_agent", "langgraph_runtime_inmem"]):
            rows = await n.ops.Threads.delete(n.conn, n.ids[0])
            assert await anext(rows) == n.ids[0]

    asyncio.run(run())
    assert str(n.ids[0]) not in n.memory.storage
    assert n.memory.get_tuple(n.saved[n.ids[-1]]).pending_writes[0][2] == {"approval": "retain"}


def test_custom_checkpointer_keeps_native_async_route(native, monkeypatch):
    import langgraph_api
    import threading

    n = native
    n.config.USE_CUSTOM_CHECKPOINTER = True
    caller = threading.get_ident()
    seen = []

    async def erase(identity):
        seen.append((identity, threading.get_ident()))

    custom = SimpleNamespace(adelete_thread=AsyncMock(side_effect=erase))
    api = SimpleNamespace(get_checkpointer=AsyncMock(return_value=custom))
    monkeypatch.setitem(sys.modules, "langgraph_api._checkpointer", api)
    monkeypatch.setattr(langgraph_api, "_checkpointer", api, raising=False)
    asyncio.run(n.ops._delete_checkpoints_for_thread(n.ids[0], n.conn))
    assert seen == [(str(n.ids[0]), caller)]
    assert str(n.ids[0]) in n.memory.storage

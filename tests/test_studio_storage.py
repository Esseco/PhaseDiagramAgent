import asyncio

from langgraph.checkpoint.memory import InMemorySaver
from phase_agent.persistence.compact_checkpoints import CompactSqliteSaver
from phase_agent.persistence.studio_storage import StudioStorage, install_memory_adapter


def checkpoint(index, value):
    return {
        "v": 4,
        "id": f"{index:04}",
        "ts": "2026-10-15T00:00:00+00:00",
        "channel_values": {"business_frame": value},
        "channel_versions": {"business_frame": index},
        "versions_seen": {},
        "pending_sends": [],
    }


def test_sql_payloads_are_collected_without_losing_latest_pending(tmp_path):
    path = str(tmp_path / "state.sqlite")
    cfg = {"configurable": {"thread_id": "t", "checkpoint_ns": ""}}
    with CompactSqliteSaver.from_conn_string(path) as saver:
        for i in range(20):
            value = {"loaded_state": {"tasks": [str(i) * 100000]}}
            cfg = saver.put(cfg, checkpoint(i, value), {}, {})
        saver.put_writes(cfg, [("__interrupt__", {"pending": "approval"})], "task")
    with CompactSqliteSaver.from_conn_string(path) as saver:
        assert saver.conn.execute("SELECT count(*) FROM checkpoint_payloads").fetchone()[0] == 8
        latest = saver.get_tuple(cfg)
        assert latest.checkpoint["channel_values"]["business_frame"] == value
        assert latest.pending_writes[0][2] == {"pending": "approval"}


def test_studio_pruning_keeps_exact_nested_parent_and_latest_interrupt():
    mem = InMemorySaver()
    manager = StudioStorage(mem)
    cfg = {"configurable": {"thread_id": "t", "checkpoint_ns": "child"}}
    for i in range(20):
        cfg = mem.put(cfg, checkpoint(i, {"facts": "x" * 10000}), {}, {"business_frame": i})
        mem.put_writes(cfg, [("fact", i)], "task")
    mem.put_writes(cfg, [("__interrupt__", "wait")], "approval")
    root = {"configurable": {"thread_id": "t", "checkpoint_ns": ""}}
    mem.put(root, checkpoint(0, {}), {"parents": {"child": "0001"}}, {"business_frame": 0})
    manager.prune()
    assert len(mem.storage["t"]["child"]) == 9
    assert mem.get_tuple(
        {"configurable": {"thread_id": "t", "checkpoint_ns": "child", "checkpoint_id": "0001"}}
    )
    assert mem.get_tuple(cfg).pending_writes[-1][2] == "wait"
    assert len(mem.blobs) == 10


def test_adapter_views_share_memory_without_file_load_and_compress(monkeypatch, tmp_path):
    from langgraph_runtime_inmem import checkpoint as runtime

    monkeypatch.chdir(tmp_path)
    original_class, original_memory = runtime.InMemorySaver, runtime.MEMORY
    monkeypatch.setattr(runtime, "MEMORY", None)
    try:
        install_memory_adapter(runtime)
        mem = runtime.InMemorySaver(serde=InMemorySaver().serde)
        runtime.MEMORY = mem
        cfg = {"configurable": {"thread_id": "t", "checkpoint_ns": ""}}
        value = {"loaded_state": {"large": "text" * 100000}}
        saved = mem.put(cfg, checkpoint(0, value), {}, {"business_frame": 0})
        kind, data = next(iter(mem.blobs.values()))
        assert kind.startswith("zlib:") and len(data) < 10000
        assert mem.get_tuple(saved).checkpoint["channel_values"]["business_frame"] == value
        for store in (mem.storage, mem.writes, mem.blobs):
            store.sync()
        before = {p.name: p.stat().st_mtime_ns for p in (tmp_path / ".langgraph_api").glob("*")}
        for store in (mem.storage, mem.writes, mem.blobs):
            store.sync()
        assert before == {
            p.name: p.stat().st_mtime_ns for p in (tmp_path / ".langgraph_api").glob("*")
        }
        from langgraph.checkpoint.memory import PersistentDict

        monkeypatch.setattr(
            PersistentDict,
            "load",
            lambda self: (_ for _ in ()).throw(AssertionError("view reload")),
        )
        view = runtime.InMemorySaver(serde=mem.serde.delegate)
        assert view.storage is mem.storage and view.blobs is mem.blobs
        assert view.get_tuple(saved).checkpoint["channel_values"]["business_frame"] == value
        view.put_writes(saved, [("__interrupt__", "keep")], "approval")
        assert mem.get_tuple(saved).pending_writes[0][2] == "keep"
        mem.writes.sync()
        # Exercise the installed runtime's actual unpack-hook factory too.
        import sys
        from types import SimpleNamespace
        import langgraph_api.serde

        monkeypatch.setattr(langgraph_api.serde, "Serializer", lambda **kw: InMemorySaver().serde)
        monkeypatch.setitem(
            sys.modules, "langgraph_api.feature_flags", SimpleNamespace(DELTA_CHANNEL_SUPPORT=False)
        )
        factory_view = runtime.Checkpointer(unpack_hook=lambda *args: None)
        assert factory_view.storage is mem.storage
        assert factory_view.get_tuple(saved).pending_writes[0][2] == "keep"
        with mem._phase_storage.reading():
            mem.put(saved, checkpoint(1, value), {}, {"business_frame": 1})
            old_revision = mem._phase_storage.saved["storage"]
            mem.storage.sync()
            assert mem._phase_storage.saved["storage"] == old_revision
        mem.storage.sync()
        assert mem._phase_storage.saved["storage"] == mem._phase_storage.revision
    finally:
        runtime.InMemorySaver = original_class
        runtime.MEMORY = original_memory


def test_flow_page_does_not_block_event_loop(monkeypatch):
    import threading
    from phase_agent.runtime import studio_runtime
    from starlette.responses import HTMLResponse

    event_thread = threading.get_ident()
    calls = []

    def render():
        calls.append(threading.get_ident())
        return HTMLResponse("facts")

    monkeypatch.setattr(studio_runtime, "_render_flow_panel", render)
    assert asyncio.run(studio_runtime.flow_panel(None)).body == b"facts"
    assert calls[0] != event_thread


def test_offline_sqlite_compaction_preserves_recovery(tmp_path):
    from phase_agent.persistence.checkpoint_maintenance import compact_sqlite

    original, target = tmp_path / "old.sqlite", tmp_path / "new.sqlite"
    cfg = {"configurable": {"thread_id": "t", "checkpoint_ns": ""}}
    from langgraph.checkpoint.sqlite import SqliteSaver

    with SqliteSaver.from_conn_string(str(original)) as saver:
        for i in range(12):
            cfg = saver.put(cfg, checkpoint(i, {"loaded_state": {"facts": "x" * 100000}}), {}, {})
        saver.put_writes(cfg, [("__interrupt__", "review")], "task")
    result = compact_sqlite(original, target)
    assert result["verified"] and result["old_checkpoints_archived"] == 4
    assert target.stat().st_size < original.stat().st_size / 5
    with CompactSqliteSaver.from_conn_string(str(target)) as saver:
        assert saver.get_tuple(cfg).pending_writes[0][2] == "review"


def test_offline_sqlite_compaction_accepts_lifecycle_retention_window(tmp_path):
    from phase_agent.persistence.checkpoint_maintenance import compact_sqlite

    original, target = tmp_path / "old.sqlite", tmp_path / "new.sqlite"
    cfg = {"configurable": {"thread_id": "t", "checkpoint_ns": ""}}
    from langgraph.checkpoint.sqlite import SqliteSaver

    with SqliteSaver.from_conn_string(str(original)) as saver:
        for i in range(9):
            cfg = saver.put(cfg, checkpoint(i, {"loaded_state": {"facts": "x" * 10000}}), {}, {})
    result = compact_sqlite(original, target, history_keep=3, namespace_keep=8)
    assert result["verified"] and result["old_checkpoints_archived"] == 6
    with CompactSqliteSaver.from_conn_string(str(target), maintain_history=False) as saver:
        assert saver.get_tuple(cfg).checkpoint["id"] == "0008"


def test_offline_pickle_compresses_without_losing_records(tmp_path):
    import pickle
    from phase_agent.persistence.checkpoint_maintenance import compact_pickle
    from phase_agent.persistence.compact_checkpoints import CompressedSerde

    mem = InMemorySaver()
    cfg = {"configurable": {"thread_id": "t", "checkpoint_ns": ""}}
    saved = mem.put(cfg, checkpoint(0, {"data": "x" * 100000}), {}, {"business_frame": 0})
    mem.put_writes(saved, [("__interrupt__", "x" * 100000)], "task")
    for name in ("storage", "writes", "blobs"):
        source, target = tmp_path / (name + ".old"), tmp_path / (name + ".new")
        with source.open("wb") as stream:
            pickle.dump(dict(getattr(mem, name)), stream)
        assert compact_pickle(source, target, name)["verified"]
        with target.open("rb") as stream:
            converted = pickle.load(stream)
        getattr(mem, name).clear()
        getattr(mem, name).update(converted)
    mem.serde = CompressedSerde(mem.serde)
    assert mem.get_tuple(saved).checkpoint["channel_values"]["business_frame"] == {
        "data": "x" * 100000
    }
    assert mem.get_tuple(saved).pending_writes[0][2] == "x" * 100000


def test_response_presentation_reads_final_state_once(tmp_path, monkeypatch):
    from phase_agent.runtime import studio_runtime
    from phase_agent.runtime.chat_application import RunWorkflowChatHandler
    import phase_agent.tools.step_runner.file_protocol as protocol

    class Handler(RunWorkflowChatHandler):
        def __call__(self, *args, **kwargs):
            return "saved facts"

    handler = Handler({"state_path": str(tmp_path / "state.json")})
    handler.config_delegate = None
    config = tmp_path / "agent_runtime.json"
    config.write_text("{}")
    monkeypatch.setenv("PHASE_AGENT_RUNTIME_CONFIG", str(config))
    monkeypatch.setattr(studio_runtime, "_handler", handler)
    calls = []

    def read(path, default=None):
        calls.append(path)
        return {"epoch": 1}

    monkeypatch.setattr(protocol, "read_json", read)
    monkeypatch.setattr(
        "phase_agent.runtime.studio_reply_presentation.format_turn_reply",
        lambda reply, *a, **kw: reply,
    )
    monkeypatch.setattr(
        "phase_agent.runtime.project_context.handler_project_context", lambda *a: {}
    )
    monkeypatch.setattr(
        "phase_agent.runtime.chat_state_presentation.format_epoch_reply",
        lambda reply, *a, **kw: reply,
    )
    reply = studio_runtime.send_project_message("status", "fake")
    body, marker, record_path = reply.partition("\n交互记录地址：")
    assert body.rstrip() == "saved facts"
    assert marker and record_path.startswith(str(tmp_path))
    assert calls == [handler.state_path]


def test_completed_namespace_retention_preserves_old_interrupt_and_parent():
    mem = InMemorySaver()
    for i in range(40):
        cfg = {"configurable": {"thread_id": "t", "checkpoint_ns": f"child:{i:04}"}}
        saved = mem.put(cfg, checkpoint(i, {}), {}, {"business_frame": i})
        if i == 0:
            mem.put_writes(saved, [("__interrupt__", "approval")], "task")
    cfg = {"configurable": {"thread_id": "t", "checkpoint_ns": ""}}
    mem.put(cfg, checkpoint(41, {}), {"parents": {"child:0001": "0001"}}, {"business_frame": 41})
    StudioStorage(mem).prune()
    assert len(mem.storage["t"]) == 35  # root + 32 recent + interrupt + reference
    assert (
        mem.get_tuple(
            {"configurable": {"thread_id": "t", "checkpoint_ns": "child:0000"}}
        ).pending_writes[0][2]
        == "approval"
    )


def test_dev_direct_mutations_keep_store_identity_and_mark_dirty(monkeypatch):
    from collections import defaultdict
    from types import SimpleNamespace
    from phase_agent.persistence.studio_storage import install_mutation_adapters

    mem = InMemorySaver()
    manager = StudioStorage(mem)
    mem._phase_storage = manager
    original_writes = mem.writes
    mem.writes[("old", "", "1")][("task", 0)] = ("task", "channel", ("null", b""), "")

    async def delete(*args, **kwargs):
        mem.writes = defaultdict(dict)

    async def copy(*args, **kwargs):
        mem.storage["copied"][""] = {}

    ops = SimpleNamespace(
        Threads=type("Threads", (), {"copy": staticmethod(copy)}),
        _delete_checkpoints_for_thread=delete,
    )
    import sys
    import langgraph_api

    config = SimpleNamespace(USE_CUSTOM_CHECKPOINTER=False)
    monkeypatch.setitem(sys.modules, "langgraph_api.config", config)
    monkeypatch.setattr(langgraph_api, "config", config, raising=False)
    install_mutation_adapters(SimpleNamespace(MEMORY=mem), ops)
    asyncio.run(ops.Threads.copy())
    asyncio.run(ops._delete_checkpoints_for_thread("old", None))
    assert mem.writes is original_writes and not mem.writes
    assert "copied" in mem.storage
    assert manager.revision == 2 and manager.readers == 0


def test_tracing_parent_without_persistence_is_not_fabricated_or_deleted():
    mem = InMemorySaver()
    cfg = {"configurable": {"thread_id": "t", "checkpoint_ns": "child"}}
    saved = mem.put(
        cfg,
        checkpoint(0, {"facts": "retained"}),
        {"parents": {"unpersisted_dialogue": "missing"}},
        {"business_frame": 0},
    )
    manager = StudioStorage(mem)
    manager.prune()
    assert manager.unavailable_parent_refs == {("t", "unpersisted_dialogue", "missing")}
    assert "unpersisted_dialogue" not in mem.storage["t"]
    assert mem.get_tuple(saved).metadata["parents"] == {"unpersisted_dialogue": "missing"}


def test_flow_reads_committed_state_without_waiting_for_busy_handler(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from phase_agent.runtime import studio_runtime

    class BusyLock:
        def __enter__(self):
            raise AssertionError("Flow page waited for a scientific turn")

        def __exit__(self, *args):
            pass

    handler = SimpleNamespace(state_path=tmp_path / "state.json", lock=BusyLock())
    monkeypatch.setattr(studio_runtime, "project_handler", lambda owner: handler)
    monkeypatch.setattr(
        "phase_agent.tools.step_runner.file_protocol.read_json", lambda *a: {"saved": True}
    )
    monkeypatch.setattr(
        "phase_agent.runtime.flow_panel.render_flow_panel",
        lambda state: "saved facts" if state["saved"] else "",
    )
    assert asyncio.run(studio_runtime.flow_panel(None)).body == b"saved facts"

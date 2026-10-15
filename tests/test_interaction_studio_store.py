from collections import defaultdict
import pickle
from pathlib import Path
import uuid
import pytest
from phase_agent.persistence.studio_checkpoint_store import (
    StudioCheckpointDict,
    migrate_studio_store,
)


def test_nested_mutations_survive_reopen_and_get_does_not_create(tmp_path):
    filename = tmp_path / "store.pckl"
    root = StudioCheckpointDict(lambda: defaultdict(dict), filename=filename)
    assert root.get("absent") is None
    assert len(root) == 0
    cid = uuid.uuid4()
    root["thread"]["ns"][cid] = (("msgpack", b"value"), ("json", b"{}"), None)
    root.sync()
    other = StudioCheckpointDict(lambda: defaultdict(dict), filename=filename)
    assert other["thread"]["ns"][cid][0][1] == b"value"
    del other["thread"]["ns"][cid]
    assert cid not in root["thread"]["ns"]


def test_leaf_cache_is_bounded_and_sync_does_not_rewrite(tmp_path):
    root = StudioCheckpointDict(filename=tmp_path / "blob.pckl")
    root.backend.max_cache_bytes = 2048
    for i in range(20):
        root[(str(i), "ns", i)] = ("binary", bytes([i]) * 1500)
    for key in root:
        assert len(root[key][1]) == 1500
    assert root.backend.cache_bytes <= 2048
    before = root.backend.conn.total_changes
    root.sync()
    root.sync()
    assert root.backend.conn.total_changes == before
    assert not Path(root.filename).exists()


def test_identical_historical_payloads_are_stored_once(tmp_path):
    root = StudioCheckpointDict(filename=tmp_path / "shared.pckl")
    for i in range(20):
        root[("thread", "facts", i)] = ("msgpack", b"large unchanged facts" * 10000)
    assert len(root) == 20
    assert root.backend.conn.execute("select count(*) from payloads").fetchone()[0] == 1
    for key in root:
        assert root[key][1] == b"large unchanged facts" * 10000


def test_legacy_migration_keeps_original_and_all_records(tmp_path):
    filename = tmp_path / "old.pckl"
    old = {
        "thread": defaultdict(
            dict,
            {"ns": {"checkpoint": (("msgpack", b"x" * 20000), ("json", b"{}"), None)}},
        )
    }
    raw = pickle.dumps(old, protocol=2)
    filename.write_bytes(raw)
    with pytest.raises(RuntimeError, match="迁移"):
        StudioCheckpointDict(filename=filename)
    report = migrate_studio_store(filename)
    assert report["verified"]
    root = StudioCheckpointDict(lambda: defaultdict(dict), filename=filename)
    assert root.materialize() == old
    assert filename.read_bytes() == raw
    assert migrate_studio_store(filename)["already_migrated"]


def test_real_saver_history_pending_writes_delete_and_resume(tmp_path):
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.checkpoint.base import empty_checkpoint

    count = 0

    def factory(*args):
        nonlocal count
        count += 1
        return StudioCheckpointDict(*args, filename=tmp_path / f"{count}.pckl")

    saver = InMemorySaver(factory=factory)
    config = {"configurable": {"thread_id": "thread", "checkpoint_ns": ""}}
    cp = empty_checkpoint()
    cp["channel_values"] = {"messages": ["preserved"], "facts": {"a": 1}}
    cp["channel_versions"] = {"messages": "1", "facts": "1"}
    saved = saver.put(
        config,
        cp,
        {"source": "input", "step": 0, "parents": {}},
        cp["channel_versions"],
    )
    saver.put_writes(saved, [("reply", "answer")], "task")
    got = saver.get_tuple(saved)
    assert got.checkpoint["channel_values"] == cp["channel_values"]
    assert got.pending_writes[0][1:] == ("reply", "answer")
    assert len(list(saver.list(config))) == 1
    saver.delete_thread("thread")
    assert list(saver.list(config)) == []


def test_replacing_outer_dictionary_keeps_lazy_pending_write_views(tmp_path):
    root = StudioCheckpointDict(dict, filename=tmp_path / "writes.pckl")
    root[("a", "ns", "cp")] = {("task", 0): ("task", "reply", ("json", b'"a"'), "path")}
    root[("b", "ns", "cp")] = {("task", 0): ("task", "reply", ("json", b'"b"'), "path")}
    replacement = {key: value for key, value in root.items() if key[0] != "a"}
    root.replace_from(replacement)
    assert list(root) == [("b", "ns", "cp")]
    assert root[("b", "ns", "cp")][("task", 0)][2][1] == b'"b"'


def test_incremental_sync_preserves_every_historical_checkpoint(tmp_path):
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.checkpoint.base import empty_checkpoint
    from phase_agent.persistence.studio_storage import StudioStorage

    count = 0

    def factory(*args):
        nonlocal count
        count += 1
        return StudioCheckpointDict(*args, filename=tmp_path / f"history{count}.pckl")

    saver = InMemorySaver(factory=factory)
    cfg = {"configurable": {"thread_id": "t", "checkpoint_ns": ""}}
    for i in range(20):
        cp = empty_checkpoint()
        cp["id"] = str(i).zfill(4)
        cfg = saver.put(cfg, cp, {"step": i}, {})
    manager = StudioStorage(saver)
    manager.changed()
    saver.storage.sync()
    assert len(list(saver.list({"configurable": {"thread_id": "t"}}))) == 20

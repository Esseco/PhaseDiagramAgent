import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from phase_agent.persistence.compact_checkpoints import CompactSqliteSaver, prune_history


def checkpoint(index):
    return {
        "v": 4,
        "id": f"{index:04}",
        "ts": "2026-10-12T00:00:00+00:00",
        "channel_values": {"business": "large-state" * 10000},
        "channel_versions": {"business": index},
        "versions_seen": {},
        "pending_sends": [],
    }


def test_reads_legacy_and_compresses_new_records(tmp_path):
    path = str(tmp_path / "graph.sqlite")
    config = {"configurable": {"thread_id": "project", "checkpoint_ns": ""}}
    with SqliteSaver.from_conn_string(path) as old:
        saved = old.put(config, checkpoint(0), {"source": "input", "step": 0}, {})
    with CompactSqliteSaver.from_conn_string(path) as saver:
        assert saver.get_tuple(saved).checkpoint == checkpoint(0)
        saved = saver.put(
            saved,
            checkpoint(1),
            {"source": "loop", "step": 1, "writes": {"business": "x" * 100000}},
            {},
        )
        saver.put_writes(saved, [("business", "pending" * 10000)], "pending-task")
        data = saver.get_tuple(saved)
        assert data.checkpoint == checkpoint(1)
        assert data.pending_writes[0][2] == "pending" * 10000
        assert data.metadata["writes"]["business"] == "x" * 100000
        assert (
            saver.conn.execute(
                "SELECT length(checkpoint) FROM checkpoints WHERE checkpoint_id='0001'"
            ).fetchone()[0]
            < 3000
        )


def test_retention_preserves_latest_and_pending_writes_per_namespace(tmp_path):
    path = str(tmp_path / "graph.sqlite")
    configs = []
    with CompactSqliteSaver.from_conn_string(path) as saver:
        for namespace in ("", "training"):
            config = {"configurable": {"thread_id": "same", "checkpoint_ns": namespace}}
            for index in range(20):
                config = saver.put(config, checkpoint(index), {"source": "loop", "step": index}, {})
                saver.put_writes(config, [("business", f"pending-{index}")], "task")
            configs.append(config)
    with CompactSqliteSaver.from_conn_string(path) as saver:
        for config in configs:
            data = saver.get_tuple(config)
            assert data.checkpoint["id"] == "0019"
            assert data.pending_writes[0][2] == "pending-19"
        assert saver.conn.execute("SELECT count(*) FROM checkpoints").fetchone()[0] == 10
        assert saver.conn.execute("SELECT count(*) FROM writes").fetchone()[0] == 10
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert prune_history(conn) == 0


def test_subgraph_retention_is_bounded_and_protects_old_interrupt(tmp_path):
    path = str(tmp_path / "subgraphs.sqlite")
    with CompactSqliteSaver.from_conn_string(path) as saver:
        for index in range(40):
            config = {
                "configurable": {"thread_id": "project", "checkpoint_ns": f"observe:{index:04}"}
            }
            saved = saver.put(config, checkpoint(index), {"source": "loop", "step": index}, {})
            if index == 0:
                saver.put_writes(saved, [("__interrupt__", "pending-approval")], "task")
        root = {"configurable": {"thread_id": "project", "checkpoint_ns": ""}}
        saver.put(root, checkpoint(0), {"source": "loop", "step": 0}, {})
    with CompactSqliteSaver.from_conn_string(path) as saver:
        assert len(list(saver.list({"configurable": {"thread_id": "project"}}))) == 34
        interrupted = saver.get_tuple(
            {"configurable": {"thread_id": "project", "checkpoint_ns": "observe:0000"}}
        )
        assert interrupted.pending_writes[0][2] == "pending-approval"
        assert saver.get_tuple(root) is not None

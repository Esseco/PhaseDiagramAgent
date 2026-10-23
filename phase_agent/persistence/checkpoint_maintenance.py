"""Offline checkpoint conversion. Call only while owning the project service lock."""

from hashlib import sha256
from pathlib import Path
import pickle
import sqlite3
from phase_agent.persistence.sqlite_paths import readonly_uri
import zlib


def compress_typed(value):
    kind, data = value
    if kind.startswith("zlib:") or len(data) < 1024:
        return value
    compressed = zlib.compress(data, 1)
    if len(compressed) >= len(data):
        return value
    if zlib.decompress(compressed) != data:
        raise ValueError("Checkpoint compression verification failed")
    return "zlib:" + kind, compressed


def compact_pickle(source, destination, role):
    """Preserve all records and typed bytes, process one large file at a time."""
    with open(source, "rb") as stream:
        value = pickle.load(stream)
    count = 0
    if role == "storage":
        for namespaces in value.values():
            for entries in namespaces.values():
                for key, (checkpoint, metadata, parent) in entries.items():
                    entries[key] = (
                        compress_typed(checkpoint),
                        compress_typed(metadata),
                        parent,
                    )
                    count += 1
    elif role == "writes":
        for entries in value.values():
            for key, (task, channel, typed, path) in entries.items():
                entries[key] = (task, channel, compress_typed(typed), path)
                count += 1
    elif role == "blobs":
        for key in value:
            value[key] = compress_typed(value[key])
            count += 1
    else:
        raise ValueError("Unknown Studio store")
    with open(destination, "wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        import os

        os.fsync(stream.fileno())
    # Fingerprint the complete converted container. A second sequential read
    # verifies file persistence without keeping both huge old stores resident.
    expected = sha256(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()
    del value
    with open(destination, "rb") as stream:
        restored = pickle.load(stream)
    actual = sha256(pickle.dumps(restored, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()
    if actual != expected:
        raise ValueError("Studio persisted checkpoint verification failed")
    return {
        "records": count,
        "before": Path(source).stat().st_size,
        "after": Path(destination).stat().st_size,
        "verified": True,
    }


def compact_sqlite(source, destination, *, preserve_all=False):
    """Migrate retained envelopes to compressed CAS, collect garbage, then vacuum."""
    from phase_agent.persistence.compact_checkpoints import (
        CompactSqliteSaver,
        collect_payloads,
        prune_history,
    )

    from contextlib import closing

    with closing(sqlite3.connect(readonly_uri(source), uri=True)) as original:
        with closing(sqlite3.connect(destination)) as target:
            original.backup(target)
    with sqlite3.connect(destination) as conn:
        if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Original checkpoint SQLite failed integrity check")
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"checkpoints", "writes"}.issubset(tables):
            raise ValueError("Unknown checkpoint database schema")
        saver = CompactSqliteSaver(conn)
        latest = list(
            conn.execute(
                "SELECT thread_id,checkpoint_ns,max(checkpoint_id) FROM checkpoints GROUP BY thread_id,checkpoint_ns"
            )
        )

        def evidence():
            result = {}
            for thread, ns, cid in latest:
                item = saver.get_tuple(
                    {
                        "configurable": {
                            "thread_id": thread,
                            "checkpoint_ns": ns,
                            "checkpoint_id": cid,
                        }
                    }
                )
                if item is None:
                    raise ValueError("Latest recovery checkpoint disappeared")
                kind, raw = saver.serde.raw_delegate.dumps_typed(
                    (item.checkpoint, item.pending_writes, item.metadata)
                )
                result[(thread, ns, cid)] = sha256(kind.encode() + raw).hexdigest()
            return result

        before = evidence()
        removed = 0 if preserve_all else prune_history(conn, limit_namespaces=False)
        for table, column in (("checkpoints", "checkpoint"), ("writes", "value")):
            # Fetch only identifiers first, never materialize all multi-GB blobs.
            for (rowid,) in list(conn.execute(f"SELECT rowid FROM {table}")):
                kind, raw = conn.execute(
                    f"SELECT type,{column} FROM {table} WHERE rowid=?", (rowid,)
                ).fetchone()
                value = saver.serde.loads_typed((kind, raw))
                new_kind, new_raw = saver.serde.dumps_typed(value)
                if preserve_all and saver.serde.loads_typed((new_kind, new_raw)) != value:
                    raise ValueError("Historical checkpoint changed during lossless conversion")
                conn.execute(
                    f"UPDATE {table} SET type=?,{column}=? WHERE rowid=?",
                    (new_kind, new_raw, rowid),
                )
        conn.commit()
        payloads_removed = collect_payloads(conn, saver.serde.delegate)
        if before != evidence():
            raise ValueError("Recovery values or pending writes changed during compaction")
        conn.execute("VACUUM")
        if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Compacted SQLite failed integrity check")
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()
    return {
        "before": Path(source).stat().st_size,
        "after": Path(destination).stat().st_size,
        "latest_recovery_points_verified": len(latest),
        "old_checkpoints_archived": removed,
        "unreachable_payloads_removed": payloads_removed,
        "verified": True,
    }

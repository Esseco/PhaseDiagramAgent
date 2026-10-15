"""Compressed, bounded recovery history; scientific audit lives in project ledgers."""

from contextlib import contextmanager, closing
import sqlite3
import zlib

from langgraph.checkpoint.sqlite import SqliteSaver


class CompressedSerde:
    def __init__(self, delegate):
        self.delegate = delegate

    def dumps_typed(self, value):
        kind, data = self.delegate.dumps_typed(value)
        if len(data) >= 1024:
            compressed = zlib.compress(data, 1)
            if len(compressed) < len(data):
                return "zlib:" + kind, compressed
        return kind, data

    def loads_typed(self, value):
        kind, data = value
        if kind.startswith("zlib:"):
            return self.delegate.loads_typed((kind[5:], zlib.decompress(data)))
        return self.delegate.loads_typed(value)


def prune_history(conn, keep=8, *, limit_namespaces=False):
    """Keep each thread/namespace's current recovery and recent pending writes."""
    if keep < 2:
        raise ValueError("At least two recovery checkpoints must be retained")
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not {"checkpoints", "writes"}.issubset(tables):
        return 0
    before = conn.execute("SELECT count(*) FROM checkpoints").fetchone()[0]
    with conn:
        if limit_namespaces:
            # Keep recent subgraph executions and every interrupted/pending one.
            # Root threads are never retired. Native recovery uses their latest IDs.
            conn.execute("""DELETE FROM checkpoints WHERE checkpoint_ns <> ''
                AND (thread_id, checkpoint_ns) IN (
                    SELECT thread_id, checkpoint_ns FROM (
                        SELECT thread_id, checkpoint_ns, max(checkpoint_id) AS latest,
                        row_number() OVER (PARTITION BY thread_id
                            ORDER BY max(checkpoint_id) DESC) AS rank
                        FROM checkpoints WHERE checkpoint_ns <> ''
                        GROUP BY thread_id, checkpoint_ns
                    ) n WHERE rank > 32 AND NOT EXISTS (
                        SELECT 1 FROM writes w WHERE w.thread_id=n.thread_id
                        AND w.checkpoint_ns=n.checkpoint_ns AND w.checkpoint_id=n.latest
                        AND (w.channel='__interrupt__' OR w.channel='__pregel_tasks'
                             OR w.channel LIKE 'branch:to:%')
                    )
                )""")
        conn.execute(
            """DELETE FROM checkpoints WHERE rowid IN (
            SELECT rowid FROM (SELECT rowid, row_number() OVER (
                PARTITION BY thread_id, checkpoint_ns ORDER BY checkpoint_id DESC
             ) AS rank, checkpoint_ns FROM checkpoints)
            WHERE rank > CASE WHEN checkpoint_ns = '' THEN ? ELSE 2 END
        )""",
            (keep,),
        )
        conn.execute("""DELETE FROM writes WHERE NOT EXISTS (
            SELECT 1 FROM checkpoints c WHERE c.thread_id=writes.thread_id
            AND c.checkpoint_ns=writes.checkpoint_ns AND c.checkpoint_id=writes.checkpoint_id
        )""")
        conn.execute("""UPDATE checkpoints SET parent_checkpoint_id=NULL
            WHERE parent_checkpoint_id IS NOT NULL AND NOT EXISTS (
                SELECT 1 FROM checkpoints p WHERE p.thread_id=checkpoints.thread_id
                AND p.checkpoint_ns=checkpoints.checkpoint_ns
                AND p.checkpoint_id=checkpoints.parent_checkpoint_id
            )""")
    return before - conn.execute("SELECT count(*) FROM checkpoints").fetchone()[0]


def collect_payloads(conn, serde):
    """Collect only unreachable CAS payloads; decode envelopes, never business state."""
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "checkpoint_payloads" not in tables:
        return 0
    from phase_agent.persistence.checkpoint_blobs import STATE_FIELDS

    reachable = set()

    def visit(value):
        if isinstance(value, dict):
            if set(value) == {STATE_FIELDS}:
                reachable.update(
                    ref["blob"] for ref in value[STATE_FIELDS].values() if "blob" in ref
                )
            else:
                for item in value.values():
                    visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    for query in (
        "SELECT type,checkpoint FROM checkpoints WHERE type LIKE 'phase-cas-v1:%'",
        "SELECT type,value FROM writes WHERE type LIKE 'phase-cas-v1:%'",
    ):
        for kind, data in conn.execute(query):
            visit(serde.loads_typed((kind[len("phase-cas-v1:") :], data)))
    with conn:
        conn.execute("CREATE TEMP TABLE IF NOT EXISTS reachable_payloads (id TEXT PRIMARY KEY)")
        conn.execute("DELETE FROM reachable_payloads")
        conn.executemany("INSERT INTO reachable_payloads VALUES (?)", ((key,) for key in reachable))
        missing = conn.execute(
            "SELECT id FROM reachable_payloads EXCEPT SELECT id FROM checkpoint_payloads LIMIT 1"
        ).fetchone()
        if missing:
            raise ValueError("Checkpoint payload missing; storage collection blocked")
        before = conn.execute("SELECT count(*) FROM checkpoint_payloads").fetchone()[0]
        conn.execute(
            "DELETE FROM checkpoint_payloads WHERE id NOT IN (SELECT id FROM reachable_payloads)"
        )
    return before - len(reachable)


class CompactSqliteSaver(SqliteSaver):
    def __init__(self, conn):
        super().__init__(conn)
        from threading import RLock
        from phase_agent.persistence.checkpoint_blobs import CheckpointBlobSerde

        self.lock = RLock()
        self.serde = CheckpointBlobSerde(CompressedSerde(self.serde), conn, self.lock)
        self.timings = {}

    def _measured(self, name, callback, *args, **kwargs):
        from time import perf_counter

        started = perf_counter()
        try:
            return callback(*args, **kwargs)
        finally:
            with self.lock:
                self.timings[name] = self.timings.get(name, 0) + perf_counter() - started

    def put(self, *args, **kwargs):
        return self._measured("checkpoint_write", super().put, *args, **kwargs)

    def put_writes(self, *args, **kwargs):
        return self._measured("checkpoint_pending_write", super().put_writes, *args, **kwargs)

    def get_tuple(self, *args, **kwargs):
        return self._measured("checkpoint_read", super().get_tuple, *args, **kwargs)

    @classmethod
    @contextmanager
    def from_conn_string(cls, conn_string):
        with closing(sqlite3.connect(conn_string, check_same_thread=False)) as conn:
            saver = cls(conn)
            try:
                yield saver
            finally:
                # Called while the caller still owns the project's existing DB lock.
                prune_history(conn, limit_namespaces=True)
                collect_payloads(conn, saver.serde.delegate)
                # Do not copy a multi-GB database on the user-facing request path.
                # Existing free pages are reused; offline storage maintenance is separate.
                from phase_agent.runtime.turn_process import record_timing

                for name, seconds in saver.timings.items():
                    record_timing(name, seconds, category="checkpoint")

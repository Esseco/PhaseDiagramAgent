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


class CompactSqliteSaver(SqliteSaver):
    def __init__(self, conn):
        super().__init__(conn)
        self.serde = CompressedSerde(self.serde)

    @classmethod
    @contextmanager
    def from_conn_string(cls, conn_string):
        with closing(sqlite3.connect(conn_string, check_same_thread=False)) as conn:
            try:
                yield cls(conn)
            finally:
                # Called while the caller still owns the project's existing DB lock.
                prune_history(conn, limit_namespaces=True)
                free = conn.execute("PRAGMA freelist_count").fetchone()[0]
                page = conn.execute("PRAGMA page_size").fetchone()[0]
                if free * page > 128 * 1024 * 1024:
                    conn.execute("VACUUM")

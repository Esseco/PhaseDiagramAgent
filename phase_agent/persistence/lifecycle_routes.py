"""Resolve the one current lifecycle in its existing checkpoint database."""

import hashlib
import re


def lifecycle_base_thread(thread, state_path):
    from phase_agent.tools.state.workflow_control import current_revision

    revision = current_revision(state_path)
    return thread + ":control:" + revision if revision else thread


def fresh_lifecycle_thread(base, invocation_id):
    return base + ":fresh:" + hashlib.sha256(str(invocation_id).encode()).hexdigest()[:16]


def _valid_thread(base, thread):
    return thread == base or re.fullmatch(re.escape(base) + r":fresh:[0-9a-f]{16}", thread)


def current_lifecycle_thread(connection, base):
    """Read routes without schema writes, including pre-route fresh checkpoints."""
    tables = {
        row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    routed = None
    if "lifecycle_routes" in tables:
        row = connection.execute(
            "SELECT active_thread FROM lifecycle_routes WHERE base_thread=?", (base,)
        ).fetchone()
        if row:
            if not _valid_thread(base, row[0]):
                raise ValueError("当前生命周期路由不属于此控制修订；未重放动作。")
            routed = row[0]
    # Older releases isolated a request but did not retain its route. Pick the
    # newest root checkpoint, never an arbitrary cache or nested observation.
    if "checkpoints" in tables:
        if (
            routed
            and not connection.execute(
                "SELECT 1 FROM checkpoints WHERE thread_id=? AND checkpoint_ns='' LIMIT 1",
                (routed,),
            ).fetchone()
        ):
            raise ValueError("当前生命周期路由缺少检查点；未重放动作。")
        prefix = base + ":fresh:"
        rows = connection.execute(
            "SELECT thread_id FROM checkpoints WHERE checkpoint_ns='' AND "
            "(thread_id=? OR substr(thread_id,1,?)=?) ORDER BY checkpoint_id DESC",
            (base, len(prefix), prefix),
        )
        for (thread,) in rows:
            if _valid_thread(base, thread):
                return thread
    return routed or base


def activate_lifecycle_thread(connection, base, thread):
    if not _valid_thread(base, thread):
        raise ValueError("invalid_lifecycle_route")
    with connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS lifecycle_routes "
            "(base_thread TEXT PRIMARY KEY, active_thread TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO lifecycle_routes(base_thread,active_thread) VALUES(?,?) "
            "ON CONFLICT(base_thread) DO UPDATE SET active_thread=excluded.active_thread",
            (base, thread),
        )

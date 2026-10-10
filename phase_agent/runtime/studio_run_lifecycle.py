"""End project-local Studio turns across service restarts; preserve science state."""

import json
import logging
from datetime import datetime, timezone
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

logger = logging.getLogger(__name__)


async def recover_studio_runs():
    """Dev-runtime adapter, called before installing the project handler.

    These are local Studio records, not scientific tasks or execution receipts.
    END clears queued graph nodes without invoking them. Fail startup if the
    installed dev runtime no longer supports this adapter.
    """
    from langgraph_runtime_inmem.database import connect
    from langgraph_runtime_inmem.ops import Threads

    async with connect() as conn:
        return await end_orphaned_turns(conn, Threads)


async def end_orphaned_turns(conn, threads):
    now = datetime.now(timezone.utc)
    affected = set()
    for run in conn.store["runs"]:
        if run["status"] in {"pending", "running"}:
            run["status"] = "interrupted"
            run["updated_at"] = now
            affected.add(run["thread_id"])
    for thread in list(conn.store["threads"]):
        if thread["status"] in {"busy", "interrupted"} or thread["thread_id"] in affected:
            await threads.State.post(
                conn,
                {"configurable": {"thread_id": str(thread["thread_id"])}},
                None,
                "__end__",
            )
            affected.add(thread["thread_id"])
    if affected:
        logger.warning("Ended %s leftover Studio conversations on startup", len(affected))
    return len(affected)


def interrupt_studio_runs(port):
    """Bounded shutdown notification; abrupt exits are repaired at startup."""
    request = Request(
        f"http://127.0.0.1:{int(port)}/runs/cancel?action=interrupt",
        data=json.dumps({"status": "all"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with build_opener(ProxyHandler({})).open(request, timeout=2) as response:
            return response.status == 204
    except HTTPError as error:
        if error.code == 404:  # No active runs is an ordinary shutdown.
            return True
        logger.warning("Studio shutdown cancellation failed: HTTP %s", error.code)
    except OSError as error:
        logger.warning("Studio shutdown cancellation unavailable: %s", type(error).__name__)
    return False


def interrupt_owned_studio(process):
    """Use the launch command's project-owned endpoint, never scan by port."""
    import psutil

    if type(getattr(process, "pid", None)) is not int:
        return False
    if type(getattr(process, "pid", None)) is not int:
        return False
    try:
        command = psutil.Process(process.pid).cmdline()
        if "phase_agent.runtime.studio_service" in command and "--port" in command:
            return interrupt_studio_runs(command[command.index("--port") + 1])
    except psutil.NoSuchProcess:
        return True
    return False

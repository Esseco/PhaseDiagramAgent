import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

from phase_agent.runtime.studio_run_lifecycle import end_orphaned_turns, interrupt_studio_runs


def test_restart_ends_orphans_preserves_success_and_never_invokes_nodes():
    runs = [
        {"thread_id": "a", "status": "running"},
        {"thread_id": "b", "status": "pending"},
        {"thread_id": "c", "status": "success"},
    ]
    records = [
        {"thread_id": "a", "status": "busy"},
        {"thread_id": "b", "status": "busy"},
        {"thread_id": "c", "status": "idle"},
        {"thread_id": "d", "status": "interrupted"},
    ]
    calls = []

    async def post(conn, config, values, as_node):
        assert values is None and as_node == "__end__"
        tid = config["configurable"]["thread_id"]
        calls.append(tid)
        next(t for t in records if t["thread_id"] == tid)["status"] = "idle"

    conn = SimpleNamespace(store={"runs": runs, "threads": records})
    ops = SimpleNamespace(State=SimpleNamespace(post=post))
    assert asyncio.run(end_orphaned_turns(conn, ops)) == 3
    assert calls == ["a", "b", "d"]
    assert [r["status"] for r in runs] == ["interrupted", "interrupted", "success"]
    assert asyncio.run(end_orphaned_turns(conn, ops)) == 0


def test_shutdown_uses_all_interrupt_and_no_proxy():
    opener = MagicMock()
    opener.open.return_value.__enter__.return_value.status = 204
    with patch("phase_agent.runtime.studio_run_lifecycle.build_opener", return_value=opener):
        assert interrupt_studio_runs(2024)
    request = opener.open.call_args.args[0]
    assert request.full_url == "http://127.0.0.1:2024/runs/cancel?action=interrupt"
    assert request.data == b'{"status": "all"}'
    assert opener.open.call_args.kwargs["timeout"] == 2


def test_shutdown_no_runs_and_unavailable_api():
    with patch("phase_agent.runtime.studio_run_lifecycle.build_opener") as factory:
        factory.return_value.open.side_effect = HTTPError("local", 404, "none", {}, None)
        assert interrupt_studio_runs(2024)
        factory.return_value.open.side_effect = OSError("offline")
        assert not interrupt_studio_runs(2024)

import asyncio
import json
import threading
from contextlib import asynccontextmanager
from types import SimpleNamespace, ModuleType
from unittest.mock import Mock, patch

import pytest
from phase_agent.runtime.startup_diagnostics import startup_timeout, failure_summary
from phase_agent.runtime.studio_run_lifecycle import recover_studio_runs


def test_recovery_runs_outside_asgi_loop_and_restores_blocking_guard(monkeypatch):
    import sys
    from blockbuster.blockbuster import blockbuster_skip

    main_thread = threading.get_ident()
    seen = []
    conn = SimpleNamespace(
        store={
            "runs": [{"thread_id": "a", "status": "running"}],
            "threads": [{"thread_id": "a", "status": "busy"}],
        }
    )

    @asynccontextmanager
    async def connect():
        yield conn

    async def post(connection, config, values, node):
        seen.append((threading.get_ident(), blockbuster_skip.get(False)))
        assert values is None and node == "__end__"
        connection.store["threads"][0]["status"] = "idle"

    database = ModuleType("langgraph_runtime_inmem.database")
    database.connect = connect
    ops = ModuleType("langgraph_runtime_inmem.ops")
    ops.Threads = SimpleNamespace(State=SimpleNamespace(post=post))
    monkeypatch.setitem(sys.modules, database.__name__, database)
    monkeypatch.setitem(sys.modules, ops.__name__, ops)

    async def run():
        assert not blockbuster_skip.get(False)
        assert await recover_studio_runs() == 1
        assert not blockbuster_skip.get(False)
        assert await recover_studio_runs() == 0

    asyncio.run(run())
    assert seen == [(seen[0][0], True)] and seen[0][0] != main_thread
    assert conn.store["runs"][0]["status"] == "interrupted"


def test_shared_timeout_accepts_custom_value_and_rejects_invalid():
    assert startup_timeout({}) == 90
    assert startup_timeout({"agent_startup_timeout_seconds": 180}) == 180
    for value in [True, 0, 601, "90"]:
        with pytest.raises(ValueError):
            startup_timeout({"agent_startup_timeout_seconds": value})


def test_failure_shows_current_attempt_root_cause_not_previous_log(tmp_path):
    config = tmp_path / "agent_runtime.json"
    log = tmp_path / "studio_start_current.log"
    log.write_text("RuntimeError: specific startup failure\n", encoding="utf-8")
    folder = tmp_path / "workflow_state/studio"
    folder.mkdir(parents=True)
    (folder / "service.json").write_text(
        json.dumps({"error": "child exited 3", "child_log": str(log)})
    )
    detail = failure_summary(config, tmp_path / "agent_start_current.log")
    assert "specific startup failure" in detail and "child exited 3" in detail
    assert "studio_start_current.log" in detail


def test_launcher_passes_same_deadline_without_launching_real_process(tmp_path):
    from phase_agent.runtime.local_project_launcher import start_agent

    runtime = tmp_path / "runtime.json"
    process = Mock()
    process.poll.return_value = None
    response = Mock(status=200)
    context = Mock()
    context.__enter__ = Mock(return_value=response)
    context.__exit__ = Mock(return_value=False)
    with (
        patch(
            "phase_agent.runtime.local_project_launcher._load_config",
            return_value={"agent_startup_timeout_seconds": 180},
        ),
        patch("phase_agent.runtime.local_project_launcher._port_is_free", return_value=True),
        patch(
            "phase_agent.runtime.local_project_launcher.local_service_tokens",
            return_value=("a" * 24, "b" * 24),
        ),
        patch("phase_agent.runtime.local_project_launcher.time.monotonic", return_value=100),
        patch(
            "phase_agent.runtime.local_project_launcher.subprocess.Popen", return_value=process
        ) as popen,
        patch("phase_agent.runtime.local_project_launcher.urlopen", return_value=context),
    ):
        start_agent(runtime)
    command = popen.call_args.args[0]
    assert command[command.index("--startup-deadline") + 1] == "280.0"
    assert "agent_start_" in popen.call_args.kwargs["stdout"].name


def test_service_uses_parent_deadline_and_preserves_failure_details(tmp_path):
    from phase_agent.runtime.studio_service import run_service

    runtime = tmp_path / "runtime.json"
    runtime.write_text(json.dumps({"agent_startup_timeout_seconds": 180}))
    args = SimpleNamespace(
        runtime_config=str(runtime),
        port=2024,
        control_port=8765,
        parent_pid=None,
        startup_deadline=110,
    )
    process = Mock()
    process.poll.return_value = None
    opener = Mock()
    opener.open.side_effect = OSError("not ready")
    with (
        patch(
            "phase_agent.runtime.studio_service.preflight",
            return_value=(runtime, tmp_path / "cli.exe"),
        ),
        patch("phase_agent.runtime.local_project_launcher.validate_project_paths"),
        patch(
            "phase_agent.runtime.project_service.prepare_studio_config",
            return_value=tmp_path / "langgraph.json",
        ),
        patch("phase_agent.runtime.project_service.write_service") as write,
        patch("phase_agent.runtime.studio_service.subprocess.Popen", return_value=process),
        patch("phase_agent.runtime.studio_service.build_opener", return_value=opener),
        patch("phase_agent.runtime.studio_service.time.monotonic", return_value=111),
        patch("phase_agent.runtime.studio_run_lifecycle.interrupt_studio_runs"),
    ):
        with pytest.raises(RuntimeError, match="180 seconds"):
            run_service(args)
    record = write.call_args.args[1]
    assert record["status"] == "failed" and record["startup_timeout_seconds"] == 180
    assert "studio_start_" in record["child_log"]
    process.terminate.assert_called_once()


def test_stale_attempt_diagnostics_are_not_used(tmp_path):
    folder = tmp_path / "workflow_state/studio"
    folder.mkdir(parents=True)
    (folder / "service.json").write_text(
        json.dumps({"launch_log": "previous.log", "error": "previous failure"})
    )
    assert "previous failure" not in failure_summary(
        tmp_path / "runtime.json", tmp_path / "new.log"
    )

import json
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest
from phase_agent.runtime import runtime_diagnostics as diagnostics
from phase_agent.runtime import turn_progress
from phase_agent.runtime.turn_process import record_turn, timed_call


def test_capacity_uses_only_metadata_and_excludes_backups(tmp_path, monkeypatch):
    workflow = tmp_path / "workflow_state"
    (workflow / "studio/.langgraph_api").mkdir(parents=True)
    (workflow / "state.json").write_bytes(b"x" * 150)
    (workflow / "studio/.langgraph_api/large.pckl").write_bytes(b"x" * 2200)
    backup = workflow / "maintenance_backups"
    backup.mkdir()
    (backup / "archive.sqlite").write_bytes(b"x" * 10000)
    monkeypatch.setattr(diagnostics, "MIB", 1)
    monkeypatch.setattr(diagnostics, "GIB", 1000)
    monkeypatch.setattr(diagnostics.shutil, "disk_usage", lambda root: SimpleNamespace(free=500))
    with monkeypatch.context() as scope:
        scope.setattr(
            Path,
            "read_text",
            lambda *a, **kw: (_ for _ in ()).throw(AssertionError("content read")),
        )
        result = diagnostics.capacity_snapshot(
            tmp_path, {"complete": True, "sampled_at_unix": 0, "total_bytes": 0}, now=60
        )
    assert result["total_bytes"] == 2350
    assert result["growth_bytes"] == 2350
    assert len(result["warnings"]) == 5
    assert result["complete"]


def test_capacity_unavailable_is_not_reported_as_healthy(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "iterdir", lambda *a: (_ for _ in ()).throw(PermissionError()))
    monkeypatch.setattr(
        diagnostics.shutil, "disk_usage", lambda *a: (_ for _ in ()).throw(OSError())
    )
    result = diagnostics.capacity_snapshot(tmp_path)
    assert not result["complete"] and result["free_bytes"] is None
    assert result["unavailable"]


def test_live_stage_clock_and_qualified_wait_reason(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(turn_progress, "monotonic", lambda: clock[0])
    config = tmp_path / "agent_runtime.json"
    turn_progress.start_turn(tmp_path, "diagnostics-test")
    try:
        token = turn_progress.start_step("diagnostics-test", "API deadline 120 seconds", "model")
        clock[0] = 45
        item = turn_progress.active_turns(config)[0]
        monkeypatch.setattr(
            diagnostics,
            "sampled_location",
            lambda tid: [{"file": "threading.py", "function": "wait", "line": 1}],
        )
        report = diagnostics.describe_wait(item)
        assert report["step_seconds"] == 45 and report["level"] == "warning"
        assert "尚不能区分" in report["cause"] and "同步信号" in report["cause"]
        assert "thread_id" not in report
        turn_progress.end_step("diagnostics-test", token)
        clock[0] = 46
        assert turn_progress.active_turns(config)[0]["step_seconds"] == 1
    finally:
        turn_progress.end_turn("diagnostics-test")


def test_stack_samples_never_include_local_values():
    secret_value = "sk-do-not-persist-this-value"
    result = diagnostics.sampled_location(threading.get_ident())
    assert result and secret_value not in json.dumps(result)
    assert all(set(row) == {"file", "function", "line"} for row in result)


def test_monitor_reuses_capacity_and_does_not_delete_or_change_state(tmp_path, monkeypatch):
    config = tmp_path / "agent_runtime.json"
    config.write_text("{}")
    workflow = tmp_path / "workflow_state"
    workflow.mkdir()
    protected = workflow / "state.json"
    protected.write_text('{"pending_approval":true}')
    clock, samples = [0.0], []
    monkeypatch.setattr(diagnostics, "monotonic", lambda: clock[0])

    def capacity(root, previous):
        samples.append(root)
        return {"warnings": [], "unavailable": [], "complete": True, "total_bytes": 10}

    monkeypatch.setattr(diagnostics, "capacity_snapshot", capacity)
    monitor = diagnostics.RuntimeMonitor(config)
    with monkeypatch.context() as scope:
        scope.setattr(
            Path,
            "unlink",
            lambda *a, **kw: (_ for _ in ()).throw(AssertionError("automatic delete")),
        )
        monitor.tick()
        clock[0] = 15
        monitor.tick()
        clock[0] = 61
        monitor.tick()
    assert len(samples) == 2
    assert protected.read_text() == '{"pending_approval":true}'
    data = json.loads(diagnostics.snapshot_path(config).read_text(encoding="utf-8"))
    assert data["automatic_actions"] == "仅观察；不重启、不删除文件、不执行计算"


def test_snapshot_write_failure_does_not_raise(tmp_path, monkeypatch):
    config = tmp_path / "agent_runtime.json"
    monitor = diagnostics.RuntimeMonitor(config)
    monkeypatch.setattr(
        Path, "write_text", lambda *a, **kw: (_ for _ in ()).throw(PermissionError())
    )
    monitor.tick()
    assert "PermissionError" in monitor.latest["diagnostic_error"]


def test_diagnostic_history_is_never_automatically_deleted(tmp_path):
    config = tmp_path / "agent_runtime.json"
    config.write_text("{}")
    history = tmp_path / "workflow_state/dialogue_process"
    history.mkdir(parents=True)
    for i in range(101):
        (history / f"{i}.json").write_text("{}")
    with record_turn(config, "only diagnostics", "fake"):
        pass
    assert len(list(history.glob("*.json"))) == 102


def test_completed_error_records_stage_type_and_http_code(tmp_path, monkeypatch):
    config = tmp_path / "agent_runtime.json"
    config.write_text("{}")
    monitor = diagnostics.RuntimeMonitor(config)
    monkeypatch.setattr(diagnostics, "_monitors", {str(config.resolve()): monitor})

    class ServiceError(Exception):
        status_code = 503

    def fail():
        raise ServiceError("sk-secret-do-not-show")

    with pytest.raises(ServiceError):
        with record_turn(config, "fake", "fake"):
            timed_call("model response", fail, category="model")
    assert monitor.last_turn["failure"] == {
        "type": "ServiceError",
        "stage": "model response",
        "http_status": 503,
    }
    monitor.tick()
    html = diagnostics.render_diagnostics(config)
    assert "HTTP 503" in html and "model response" in html
    assert "sk-secret-do-not-show" not in json.dumps(monitor.latest)


def test_saved_snapshot_is_marked_historical_and_escaped(tmp_path):
    config = tmp_path / "agent_runtime.json"
    path = diagnostics.snapshot_path(config)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"capacity": {"warnings": ["<script>alert(1)</script>"]}}))
    html = diagnostics.render_diagnostics(config)
    assert "已保存快照" in html and "<script>" not in html


def test_monitor_start_stop_is_bounded_and_never_starts_project_services(tmp_path):
    config = tmp_path / "agent_runtime.json"
    monitor = diagnostics.start_monitor(config)
    diagnostics.stop_monitor(config, monitor)
    assert not monitor.thread.is_alive()
    assert str(config.resolve()) not in diagnostics._monitors
    data = json.loads(diagnostics.snapshot_path(config).read_text(encoding="utf-8"))
    assert data["service_status"] == "stopped" and data["startup"] is None


def test_startup_error_is_recorded_without_retry_or_restart(tmp_path, monkeypatch):
    import asyncio
    from contextlib import asynccontextmanager
    from phase_agent.runtime import studio_runtime

    config = tmp_path / "agent_runtime.json"
    config.write_text("{}")
    monkeypatch.setenv("PHASE_AGENT_RUNTIME_CONFIG", str(config))
    attempts = []

    @asynccontextmanager
    async def failing(app):
        attempts.append("initialize")
        diagnostics.service_stage(config, "加载检查点")
        raise FileNotFoundError("missing")
        yield

    monkeypatch.setattr(studio_runtime, "_project_lifespan", failing)

    async def run():
        async with studio_runtime.lifespan(None):
            pass

    with pytest.raises(FileNotFoundError):
        asyncio.run(run())
    assert attempts == ["initialize"]
    data = diagnostics.get_snapshot(config)
    assert data["service_status"] == "failed"
    assert data["service_failure"] == {"stage": "加载检查点", "type": "FileNotFoundError"}
    assert not diagnostics._monitors


def test_stale_snapshot_does_not_claim_current_health(tmp_path, monkeypatch):
    from datetime import datetime, timezone

    config = tmp_path / "agent_runtime.json"
    monitor = diagnostics.RuntimeMonitor(config)
    monitor.latest = {"updated_at": datetime.fromtimestamp(0, timezone.utc).isoformat()}
    monkeypatch.setattr(diagnostics, "_monitors", {str(config.resolve()): monitor})
    monkeypatch.setattr(diagnostics, "time", lambda: 60)
    assert diagnostics.get_snapshot(config)["stale"]
    assert "无法确认当前状态" in diagnostics.render_diagnostics(config)


def test_bad_saved_json_is_not_a_business_failure(tmp_path):
    config = tmp_path / "agent_runtime.json"
    path = diagnostics.snapshot_path(config)
    path.parent.mkdir(parents=True)
    path.write_text("[]")
    assert diagnostics.get_snapshot(config) == {}
    assert diagnostics.diagnostic_notice(config) == ""


def test_idle_process_page_refreshes_only_with_live_monitor(tmp_path):
    from phase_agent.runtime.turn_process import render_process_panel

    config = tmp_path / "agent_runtime.json"
    monitor = diagnostics.start_monitor(config)
    try:
        assert diagnostics.monitoring_active(config)
        assert 'content="30"' in render_process_panel(config)
    finally:
        diagnostics.stop_monitor(config, monitor)
    assert 'http-equiv="refresh"' not in render_process_panel(config)

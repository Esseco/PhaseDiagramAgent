import json
import os
import pytest
from phase_agent.agents.strategic import codex_team
from phase_agent.runtime.turn_process import capture_read_metrics
from phase_agent.tools.step_runner.file_protocol import write_json
from phase_agent.persistence.state.field_index import open_indexed_state
from tests.test_codex_native_team import make_control, team


@pytest.fixture(autouse=True)
def clear_cache():
    codex_team._READ_CACHE.clear()
    yield
    codex_team._READ_CACHE.clear()


def test_repeat_uses_strong_binding_but_no_redecode_or_evidence_build(tmp_path, monkeypatch):
    control, _ = make_control(tmp_path)
    write_json(
        control.state_path,
        {"tasks": [{"task_id": "t", "status": "completed"}], "event_history": ["x" * 10000]},
    )
    t = team(control)
    with capture_read_metrics() as first:
        expected = t._context()
    from phase_agent.evidence import build_decision_context as evidence

    monkeypatch.setattr(
        evidence, "build_decision_context", lambda *a, **k: pytest.fail("derived evidence rebuilt")
    )
    with capture_read_metrics() as repeat:
        actual = t._context()
    assert actual == expected
    assert first["counts"]["field_read"] > 0
    assert repeat["counts"].get("field_read", 0) == 0
    assert repeat["counts"].get("full_state_parse", 0) == 0
    assert repeat["counts"]["strong_file_validation"] == 2
    assert repeat["counts"]["derived_cache_hit"] == 1
    actual["team_state"]["tasks"][0]["status"] = "mutated"
    assert t._context() == expected


@pytest.mark.parametrize("change", ["state", "control", "config"])
def test_changed_sources_never_reuse_old_context(tmp_path, change):
    control, context = make_control(tmp_path)
    write_json(control.state_path, {"tasks": [], "active_model_version": "m1"})
    t = team(control)
    first = t._context()
    if change == "state":
        stat = control.state_path.stat()
        data = control.state_path.read_bytes().replace(b"m1", b"m2")
        control.state_path.write_bytes(data)
        os.utime(control.state_path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    elif change == "control":
        control.state_path.with_name("state.json.control.json").write_text(
            json.dumps(
                {
                    "state_path": str(control.state_path.resolve()),
                    "revision": "r2",
                    "binding": {"active_model_version": "m1"},
                    "patch": {},
                }
            )
        )
    else:
        context["config_version"] = "c2"
    with capture_read_metrics() as counters:
        result = t._context()
    assert counters["counts"].get("derived_cache_hit", 0) == 0
    assert (
        result["team_source_identity"] != first["team_source_identity"]
        or result["config_version"] != first["config_version"]
    )


def test_cache_is_bounded_and_expiration_releases_bytes(monkeypatch):
    monkeypatch.setattr(codex_team, "_READ_CACHE_LIMIT", 32)
    codex_team._save_cached("context", "one", {"small": True})
    assert codex_team._read_cached("context", "one") == {"small": True}
    codex_team._save_cached("context", "two", {"large": "x" * 64})
    assert "context" not in codex_team._READ_CACHE
    codex_team._save_cached("context", "one", {"small": True})
    monkeypatch.setattr(codex_team, "monotonic", lambda: 10**20)
    assert codex_team._read_cached("context", "one") is None
    assert not codex_team._READ_CACHE


@pytest.mark.parametrize("kind", ["missing", "corrupt", "stale"])
def test_index_fallback_reports_reason(tmp_path, kind):
    path = tmp_path / "state.json"
    write_json(path, {"tasks": []})
    index = path.with_name("state.json.index.json")
    if kind == "missing":
        index.unlink()
    elif kind == "corrupt":
        index.write_text("broken")
    else:
        path.write_text('{"tasks":[],"new":true}')
    with capture_read_metrics() as counters:
        assert open_indexed_state(path) is None
    assert counters["counts"]["index_fallback"] == 1
    assert counters["details"][0]["reason"]


def test_concurrent_change_rejected_on_derived_cache_hit(tmp_path, monkeypatch):
    control, _ = make_control(tmp_path)
    write_json(control.state_path, {"tasks": []})
    t = team(control)
    t._context()
    original = control.codex_context

    def change():
        value = original()
        write_json(control.state_path, {"tasks": [], "changed": True})
        return value

    monkeypatch.setattr(control, "codex_context", change)
    with pytest.raises(ValueError, match="sources_changed"):
        t._context()


def test_mcp_overview_does_not_construct_full_runtime(tmp_path, monkeypatch):
    from phase_agent.runtime.codex.mcp import CodexMcpServer
    from phase_agent.runtime.codex.offline import CodexOffline

    state = tmp_path / "state.json"
    write_json(state, {"tasks": []})
    runtime = tmp_path / "agent_runtime.json"
    runtime.write_text(
        json.dumps({"state_path": str(state), "ledger_path": str(tmp_path / "ledger.json")})
    )
    monkeypatch.setattr(
        CodexOffline,
        "__init__",
        lambda *a, **k: pytest.fail("full runtime constructed for overview"),
    )
    server = CodexMcpServer(runtime_config=runtime)
    with capture_read_metrics() as metrics:
        first = server._call_tool("phase_overview", {})
        second = server._call_tool("phase_overview", {})
    assert first["task_counts"] == second["task_counts"]
    assert second["cache_status"] == "hit"
    assert server._needs_runtime
    assert metrics["counts"].get("full_state_parse", 0) == 0


def test_mcp_upgrades_once_only_when_needed(tmp_path, monkeypatch):
    from phase_agent.runtime.codex.mcp import CodexMcpServer
    from phase_agent.runtime.codex.offline import CodexOffline

    state = tmp_path / "state.json"
    write_json(state, {"tasks": []})
    runtime = tmp_path / "agent_runtime.json"
    runtime.write_text(json.dumps({"state_path": str(state)}))
    server = CodexMcpServer(runtime_config=runtime)
    calls = []

    class Runtime:
        def __init__(self, path):
            calls.append(path)

    server._codex_factory = Runtime
    server._require_project()
    server._require_project()
    assert calls == [str(runtime)]
    assert not server._needs_runtime


def test_control_cache_checks_content_even_with_preserved_mtime(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from threading import RLock
    from phase_agent.runtime.local_agent_control import LocalAgentControl
    from phase_agent.runtime.codex import context as module

    path = tmp_path / "state.json"
    write_json(path, {"active_model_version": "m1"})
    control = LocalAgentControl(SimpleNamespace(state_path=path, lock=RLock(), workflow_kwargs={}))
    calls = []

    def build(handler):
        calls.append(1)
        return json.loads(handler.state_path.read_text())

    monkeypatch.setattr(module, "build_codex_context", build)
    assert control.codex_context()["active_model_version"] == "m1"
    saved = path.stat()
    path.write_bytes(path.read_bytes().replace(b"m1", b"m2"))
    os.utime(path, ns=(saved.st_atime_ns, saved.st_mtime_ns))
    assert control.codex_context()["active_model_version"] == "m2"
    assert len(calls) == 2


def test_cache_bypasses_numeric_keys_to_preserve_historical_hash_semantics():
    payload = {"legacy": {10: "ten", 2: "two"}}
    codex_team._save_cached("context", ("project",), payload)
    assert "context" not in codex_team._READ_CACHE
    assert list(payload["legacy"]) == [10, 2]

import importlib.util
import json
from http.client import HTTPConnection
from pathlib import Path
import sys
import threading
from unittest.mock import Mock

import pytest

from phase_agent.interface_compatibility import call_with_optimizations
from phase_agent.runtime.loaded_code import loaded_code_report


def _load(name, path, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_persistent_interpreter_keeps_old_objects_across_source_update(tmp_path, monkeypatch):
    name = "persistent_provider"
    path = tmp_path / (name + ".py")
    old = (
        "from dataclasses import dataclass\n@dataclass\nclass Runner:\n"
        " def prepare(self, state):\n  return dict(state, old_interface=True)\n"
    )
    path.write_text(old)
    provider = _load(name, path, monkeypatch)
    runner = provider.Runner()
    before = loaded_code_report(module_names=[name], source_root=tmp_path)
    assert before["status"] == "checked_code_matches"
    # Same interpreter, same class and instance: no reload or replacement.
    path.write_text(old.replace("prepare(self, state)", "prepare(self, state, *, _owned=False)"))
    caller_path = tmp_path / "lazy_caller.py"
    caller_path.write_text(
        "from phase_agent.interface_compatibility import call_with_optimizations\n"
        "def prepare(runner, state):\n"
        " return call_with_optimizations(runner.prepare, state, _optimizations={'_owned':True})\n"
    )
    caller = _load("lazy_caller", caller_path, monkeypatch)
    with pytest.raises(TypeError, match="_owned"):
        runner.prepare({}, _owned=True)
    assert caller.prepare(runner, {}) == {"old_interface": True}
    assert provider.Runner is type(runner)
    after = loaded_code_report(module_names=[name], source_root=tmp_path)
    assert after["status"] == "manual_reload_required"
    assert after["modules"][0]["changed_functions"] == ["Runner.prepare"]
    assert runner.prepare({}) == {"old_interface": True}


def test_shadow_source_path_is_reported_without_importing_or_reloading(tmp_path, monkeypatch):
    path = tmp_path / "shadow.py"
    path.write_text("def function():\n return 1\n")
    _load("wrong_location", path, monkeypatch)
    result = loaded_code_report(module_names=["wrong_location", "not_loaded"], source_root=tmp_path)
    assert result["modules"][0]["status"] == "unexpected_source_path"
    assert result["modules"][1]["status"] == "not_loaded"
    assert "not_loaded" not in sys.modules


def test_scientific_options_are_never_silently_removed():
    calls = []

    def old_function(state):
        calls.append(state)

    with pytest.raises(ValueError, match="Only performance"):
        call_with_optimizations(old_function, {}, _optimizations={"include_dft": False})
    with pytest.raises(TypeError, match="include_dft"):
        call_with_optimizations(
            old_function, {}, include_dft=False, _optimizations={"_owned": True}
        )
    assert calls == []


def test_new_snapshot_caller_accepts_held_old_memory_and_migration_functions(monkeypatch):
    from copy import deepcopy
    from phase_agent.tools.state import state_manager

    migration = state_manager.migrate_persisted_state
    memory = state_manager.update_short_term_memory
    calls = []

    def old_migration(state):
        calls.append("migration")
        return migration(state)

    def old_memory(state, snapshot):
        calls.append("memory")
        return memory(state, snapshot)

    monkeypatch.setattr(state_manager, "migrate_persisted_state", old_migration)
    monkeypatch.setattr(state_manager, "update_short_term_memory", old_memory)
    source = {"status": "ready", "tasks": []}
    original = deepcopy(source)
    result = state_manager.update_state_snapshot(source, config_version="v", _owned=True)
    assert source == original
    assert result["current_state_snapshot"]["config_version"] == "v"
    assert calls == ["migration", "memory"]


def test_stale_code_blocks_handler_and_returns_diagnostic_markdown(tmp_path, monkeypatch):
    from phase_agent.runtime import loaded_code, studio_runtime

    owner = Mock()
    monkeypatch.setattr(studio_runtime, "_handler", owner)
    config = tmp_path / "agent_runtime.json"
    config.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("PHASE_AGENT_RUNTIME_CONFIG", str(config))
    monkeypatch.setattr(
        loaded_code,
        "loaded_code_report",
        lambda: {
            "status": "manual_reload_required",
            "modules": [{"module": "old_module", "status": "source_changed"}],
        },
    )
    reply = studio_runtime.send_project_message("同意", "original-thread")
    assert "手动重新启动" in reply and "本轮未执行" in reply
    report = Path(reply.split("交互记录地址：", 1)[1])
    assert report.is_file() and "old_module" in report.read_text(encoding="utf-8")
    owner.assert_not_called()
    assert config.read_text(encoding="utf-8") == "{}"


def test_runtime_code_endpoint_is_authenticated_and_read_only(monkeypatch):
    from phase_agent.runtime.agent_api import create_server
    from phase_agent.runtime import loaded_code

    report = {"status": "manual_reload_required", "modules": []}
    monkeypatch.setattr(loaded_code, "loaded_code_report", lambda: report)
    handler = Mock(side_effect=AssertionError("must not invoke project"))
    server = create_server(handler, api_key="a" * 24, control_api_key="b" * 24, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = HTTPConnection(*server.server_address, timeout=5)
    try:
        connection.request("GET", "/phase/runtime-code")
        response = connection.getresponse()
        assert response.status == 401
        response.read()
        connection.request(
            "GET", "/phase/runtime-code", headers={"Authorization": "Bearer " + "b" * 24}
        )
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read()) == report
        handler.assert_not_called()
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

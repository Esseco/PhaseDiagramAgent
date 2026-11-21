import json
import subprocess
import sys

import pytest

from phase_agent.runtime import runtime_composition as runtime


def test_unchanged_startup_state_is_reused_without_copy(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"tasks": [{"id": 1}]}), encoding="utf-8")
    state, identity = runtime._read_runtime_state(path)
    assert runtime._refresh_runtime_state(path, path, state, identity, reusable=True) is state
    assert runtime._refresh_runtime_state(path, path, state, identity, reusable=False) is None
    assert runtime._refresh_runtime_state(tmp_path / "other", path, state, identity, reusable=True) is None


def test_separate_process_replacement_invalidates_reuse(tmp_path):
    path = tmp_path / "state.json"
    path.write_text('{"value": 1}', encoding="utf-8")
    state, identity = runtime._read_runtime_state(path)
    code = "import os,sys; from pathlib import Path; p=Path(sys.argv[1]); q=p.with_suffix('.new'); q.write_text('{\"value\": 2}', encoding='utf-8'); os.replace(q,p)"
    subprocess.run([sys.executable, "-c", code, str(path)], check=True)
    assert runtime._refresh_runtime_state(path, path, state, identity, reusable=True) is None
    refreshed, _ = runtime._read_runtime_state(path)
    assert refreshed == {"value": 2}


def test_change_during_read_fails_closed(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    path.write_text('{"value": 1}', encoding="utf-8")
    original = runtime._load_json_object
    def read_then_replace(*args):
        result = original(*args)
        replacement = tmp_path / "replacement.json"
        replacement.write_text('{"value": 2}', encoding="utf-8")
        replacement.replace(path)
        return result
    monkeypatch.setattr(runtime, "_load_json_object", read_then_replace)
    with pytest.raises(ValueError, match="runtime_state_changed_during_read"):
        runtime._read_runtime_state(path)

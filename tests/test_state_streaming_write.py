import json

import pytest

from phase_agent.tools.step_runner import file_protocol


def test_streamed_state_has_same_bytes_and_noop_does_not_replace(tmp_path):
    path = tmp_path / "state.json"
    state = {"b": [1, 2], "a": "中文"}
    file_protocol.write_json(path, state)
    expected = json.dumps(state, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":")) + "\n"
    assert path.read_bytes() == expected.encode("utf-8")
    before = path.stat().st_mtime_ns
    file_protocol.write_json(path, state)
    assert path.stat().st_mtime_ns == before


def test_interrupted_state_write_keeps_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    file_protocol.write_json(path, {"old": 1})
    before = path.read_bytes()
    def fail(*args, **kwargs):
        raise MemoryError()
    monkeypatch.setattr(file_protocol.json.JSONEncoder, "iterencode", fail)
    with pytest.raises(MemoryError):
        file_protocol.write_json(path, {"new": 1})
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))

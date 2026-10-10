import pytest

from phase_agent.tools.step_runner.file_protocol import write_json
from phase_agent.runtime.runtime_session import load_or_initialize_runtime_session


def test_nonempty_state_without_confirmed_config_cannot_be_replaced(tmp_path):
    path = tmp_path / "session.json"
    with pytest.raises(ValueError, match="有运行数据"):
        load_or_initialize_runtime_session({"tasks": [{"task_id": "T1"}]},
            state_path=tmp_path / "state.json", resolved_session=path, workspace_defaults={})
    assert not path.exists()


def test_confirmed_state_is_recovered_without_creating_draft(tmp_path):
    path = tmp_path / "session.json"
    state = {"confirmed_config": {"preserved": 17}, "confirmed_config_version": "cfg-1"}
    session = load_or_initialize_runtime_session(state, state_path=tmp_path / "state.json",
                                                resolved_session=path, workspace_defaults={})
    assert session["status"] == "confirmed"
    assert session["confirmed_snapshot"]["config"] == state["confirmed_config"]
    assert not path.exists()


def test_existing_confirmed_session_is_not_overwritten(tmp_path):
    path = tmp_path / "session.json"
    write_json(path, {"status": "confirmed", "config": {"preserved": 17}})
    original = path.read_bytes()
    session = load_or_initialize_runtime_session({}, state_path=tmp_path / "state.json",
                                                resolved_session=path, workspace_defaults={})
    assert session["config"]["preserved"] == 17
    assert path.read_bytes() == original

from types import SimpleNamespace
from unittest.mock import patch
import json

from phase_agent.graphs.dialogue.nodes import read_project
from phase_agent.runtime.turn_process import record_turn, read_presentation_state, _current


def test_real_readonly_node_uses_one_parsed_state_and_releases_turn_cache(tmp_path):
    path = tmp_path / "workflow_state/state.json"
    path.parent.mkdir()
    path.write_text('{"tasks":[],"active_model_version":"m1"}')
    config = tmp_path / "agent_runtime.json"
    config.write_text("{}")
    handler = SimpleNamespace(state_path=path, config_delegate=None, conversation_id=None)
    runtime = SimpleNamespace(
        context=SimpleNamespace(
            handler=handler,
            messages=[{"role": "user", "content": "现在什么状态"}],
            conversation_id="thread",
        )
    )
    with record_turn(config, "现在什么状态", "thread"):
        with patch(
            "phase_agent.tools.step_runner.file_protocol.json.loads", wraps=json.loads
        ) as loads:
            answer = read_project({}, runtime)
            assert answer["operation"] == "done" and handler.last_turn_outcome == "answered"
            assert read_presentation_state(path)["active_model_version"] == "m1"
            assert loads.call_count == 0
            assert (
                "_initial_state" not in _current.get()
                and "_presentation_state" not in _current.get()
            )


def test_unused_presentation_cache_is_not_written_to_process_log(tmp_path):
    from phase_agent.runtime.turn_process import remember_presentation_state, presentation_signature

    config = tmp_path / "agent_runtime.json"
    config.write_text("{}")
    path = tmp_path / "workflow_state/state.json"
    with record_turn(config, "状态", "thread"):
        remember_presentation_state(
            path, {"large_private_snapshot": ["x"]}, presentation_signature(path)
        )
    record = json.loads(
        next((tmp_path / "workflow_state/dialogue_process").glob("*.json")).read_text(
            encoding="utf-8"
        )
    )
    assert "_presentation_state" not in record and "large_private_snapshot" not in str(record)

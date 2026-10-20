from pathlib import Path
import json
from types import SimpleNamespace
from unittest.mock import Mock
import threading
import pytest

from phase_agent.tools.local.identify_rerun_plan import _input_file_count
from phase_agent.graphs.dialogue.maintenance.rerun import handle
from phase_agent.runtime.turn_process import record_turn, timed_call, render_process_panel
from phase_agent.runtime.turn_progress import active_turns
from phase_agent.runtime.studio_reply_presentation import format_turn_reply
from phase_agent.runtime.review_file_presentation import file_location_lines


def test_thousand_missing_inputs_build_index_once(monkeypatch):
    import phase_agent.tools.remote.batch_runner as runner

    index = Mock(return_value={})
    monkeypatch.setattr(runner, "_task_directories_by_id", index)
    monkeypatch.setattr(Path, "is_file", lambda self: False)
    tasks = [{"task_id": str(i), "input_path": "missing", "batch_id": "b"} for i in range(1200)]
    assert _input_file_count({}, tasks) == 0
    index.assert_called_once()


def test_existing_direct_inputs_never_build_index(monkeypatch):
    import phase_agent.tools.remote.batch_runner as runner

    index = Mock(side_effect=AssertionError("unnecessary scan"))
    monkeypatch.setattr(runner, "_task_directories_by_id", index)
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    assert _input_file_count({}, [{"input_path": "existing"}] * 1200) == 1200
    index.assert_not_called()


@pytest.mark.parametrize(
    "input_text",
    ["重新生成branch", "重新生成当前轮DFT输入", "重新生成上一轮branch", "重新读取当前轮MC结果"],
)
def test_natural_requests_reach_normal_decision_pipeline(input_text, monkeypatch):
    import phase_agent.tools.local.identify_rerun_plan as module

    identify = Mock(side_effect=AssertionError("historical interception"))
    monkeypatch.setattr(module, "identify_rerun_plan", identify)
    state = {"active_model_version": "new", "pending_execution_policies": {"p": {}}}
    assert handle(SimpleNamespace(), input_text, state, [], None) is None
    identify.assert_not_called()
    assert state["active_model_version"] == "new"


def test_pending_card_cannot_hide_actual_answer(monkeypatch):
    import phase_agent.runtime.plan_queries as queries

    monkeypatch.setattr(queries, "pending_plan_reply", lambda state: "旧方案待确认")
    state = {
        "pending_execution_policies": {
            "p": {"agent_proposal": {"recommended_action": "generate_branches"}}
        }
    }
    text = format_turn_reply("本次没有生成新 branch；只核对了已有记录。", state)
    assert "本次没有生成新 branch" in text and "旧方案待确认" in text


def test_nested_progress_visible_and_cleaned_after_failure(tmp_path):
    config = tmp_path / "agent_runtime.json"
    config.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        with record_turn(config, "test", "thread"):

            def outer():
                assert active_turns(config)[0]["label"] == "准备 Agent 决策"

                def model():
                    page = render_process_panel(config)
                    assert "等待模型响应" in page and 'http-equiv="refresh"' in page
                    assert active_turns(tmp_path / "other" / "config.json") == []
                    raise ValueError("fake failure")

                timed_call("model", model, category="model")

            timed_call("decide", outer)
    assert active_turns(config) == []
    page = render_process_panel(config)
    assert 'http-equiv="refresh"' not in page


def test_process_view_does_not_wait_for_message_lock(tmp_path):
    import phase_agent.runtime.studio_runtime as runtime

    config = tmp_path / "agent_runtime.json"
    config.write_text("{}", encoding="utf-8")
    done = threading.Event()
    pages = []

    def render():
        pages.append(runtime._render_saved_process(config))
        done.set()

    with runtime._message_lock:
        worker = threading.Thread(target=render, daemon=True)
        worker.start()
        assert done.wait(2), "process view blocked behind the busy message worker"
    worker.join(2)
    assert "本轮过程" in pages[0]


def test_review_location_shows_complete_address(tmp_path):
    path = tmp_path / "branch.json"
    text = "\n".join(file_location_lines("Branch", path))
    assert str(tmp_path) in text and "branch.json" in text
    assert str(path) in text and "文件夹：" not in text and "文件名：" not in text
    assert "](" not in text

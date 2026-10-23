import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from phase_agent.runtime.turn_process import record_turn, timed_call


def project(tmp_path, monkeypatch):
    path = tmp_path / "agent_runtime.json"
    path.write_text(json.dumps({"state_path": "workflow_state/state.json"}), encoding="utf-8")
    state = tmp_path / "workflow_state/state.json"
    state.parent.mkdir()
    state.write_text('{"tasks": [], "status": "ready"}', encoding="utf-8")
    monkeypatch.setenv("PHASE_AGENT_RUNTIME_CONFIG", str(path))
    return path, state


def test_failure_markdown_preserves_cause_and_locations_without_credentials(tmp_path, monkeypatch):
    config, state = project(tmp_path, monkeypatch)
    before = state.read_bytes()

    def broken_contract():
        raise ValueError("missing candidate_id; Authorization: Bearer sk-examplePRIVATE")

    with pytest.raises(ValueError, match="missing candidate_id"):
        with record_turn(config, "test message", "test-thread"):
            timed_call("proposal_contract", broken_contract)
    entry = json.loads(
        next((tmp_path / "workflow_state/dialogue_process").glob("*.json")).read_text(
            encoding="utf-8"
        )
    )
    failure = entry["failure_summary"]
    assert "missing candidate_id" in failure["message"]
    assert failure["stage"] == "proposal_contract"
    assert any(
        row["function"] == "broken_contract" and row["line"] > 0 for row in failure["locations"]
    )
    report = Path(entry["interaction_report"]).read_text(encoding="utf-8")
    assert "故障详情" in report and "broken_contract" in report and "missing candidate_id" in report
    assert "sk-examplePRIVATE" not in report
    assert "sk-examplePRIVATE" not in json.dumps(entry)
    assert state.read_bytes() == before


def test_known_code_mismatch_returns_report_and_never_invokes_handler(tmp_path, monkeypatch):
    from phase_agent.runtime import studio_runtime, loaded_code

    _, state = project(tmp_path, monkeypatch)
    before = state.read_bytes()
    handler = Mock(side_effect=AssertionError("must not invoke project handler"))
    monkeypatch.setattr(studio_runtime, "_handler", handler)
    monkeypatch.setattr(
        loaded_code,
        "loaded_code_report",
        lambda: {
            "status": "manual_reload_required",
            "modules": [
                {
                    "module": "old_module",
                    "status": "source_changed",
                    "changed_functions": ["action"],
                }
            ],
        },
    )
    reply = studio_runtime.send_project_message("同意", "test-thread")
    assert "本轮未执行" in reply and "手动重新启动" in reply
    assert "交互记录地址：" in reply
    report = Path(reply.split("交互记录地址：", 1)[1]).read_text(encoding="utf-8")
    assert "manual_reload_required" in report and "old_module" in report
    handler.assert_not_called()
    assert state.read_bytes() == before


def test_unexpected_guard_error_is_not_hidden_and_is_recorded(tmp_path, monkeypatch):
    from phase_agent.runtime import studio_runtime, loaded_code

    project(tmp_path, monkeypatch)
    handler = Mock(side_effect=AssertionError("must not invoke project handler"))
    monkeypatch.setattr(studio_runtime, "_handler", handler)
    monkeypatch.setattr(
        loaded_code,
        "assert_loaded_code_current",
        Mock(side_effect=RuntimeError("unexpected guard failure")),
    )
    with pytest.raises(RuntimeError, match="unexpected guard failure"):
        studio_runtime.send_project_message("test", "test-thread")
    reports = list((tmp_path / "interaction_records").rglob("*.md"))
    assert len(reports) == 1
    assert "unexpected guard failure" in reports[0].read_text(encoding="utf-8")
    handler.assert_not_called()

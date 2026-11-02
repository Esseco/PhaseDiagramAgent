"""A read-only query shows a real saved failure; it must never perform recovery."""

from copy import deepcopy
import pytest

from phase_agent.tools.step_runner.file_protocol import read_json, write_json
from tests.test_skill_recovery_boundaries import recovery_handler
from tests.test_unified_react_dialogue import make_handler


@pytest.mark.parametrize("message", ["当前状态", "查看方案"])
def test_read_only_shortcut_does_not_invent_or_trigger_recovery(tmp_path, monkeypatch, message):
    handler, requests, state = recovery_handler(
        tmp_path, monkeypatch, {"kind": "answer", "answer": "unused"}
    )
    reply = handler([{"role": "user", "content": message}], conversation_id="c")
    assert "结果回收失败" not in reply
    assert read_json(handler.state_path, {}) == state and requests == []


@pytest.mark.parametrize("message", ["当前状态", "查看方案"])
def test_read_only_shortcut_keeps_actual_recovery_error_after_reload(
    tmp_path, monkeypatch, message
):
    handler, _, state = recovery_handler(
        tmp_path, monkeypatch, {"kind": "answer", "answer": "只读解释"}
    )
    reply = handler([{"role": "user", "content": "解释回收问题"}], conversation_id="c")
    assert "synthetic invalid return" in reply
    reloaded = make_handler(tmp_path, lambda p: pytest.fail("Query must not call model"))
    reply = reloaded([{"role": "user", "content": message}], conversation_id="c")
    assert "结果回收失败" in reply and "synthetic invalid return" in reply
    assert read_json(handler.state_path, {}) == state


def test_approved_failed_recovery_preserves_pending_and_never_acknowledges(tmp_path, monkeypatch):
    from phase_agent.tools.workflows import recover_saved_results as recovery
    from phase_agent.graphs.dialogue import saved_control

    handler, _, state = recovery_handler(
        tmp_path, monkeypatch, {"kind": "answer", "answer": "失败原因"}
    )
    state["pending_execution_policies"] = {
        "old-plan": {"agent_proposal": {"raw_action": {"tool": "check_convergence"}}}
    }
    write_json(handler.state_path, state)
    monkeypatch.setattr(
        recovery, "recovery_candidate", lambda *a, **kw: {"pending": True, "fingerprint": "return1"}
    )
    monkeypatch.setattr(saved_control, "semantic_recovery_intent", lambda *a, **kw: "approve")
    reply = handler([{"role": "user", "content": "已回传"}], conversation_id="c")
    assert "synthetic invalid return" in reply
    assert read_json(handler.state_path, {}) == state
    assert not read_json(handler.state_path, {}).get("recovery_acknowledgement")
    assert not read_json(handler.state_path, {}).get("post_recovery_next_step_pending")


def test_real_success_clears_current_error_but_noop_does_not(tmp_path, monkeypatch):
    from phase_agent.tools.workflows import recover_saved_results as recovery
    from phase_agent.runtime.recovery_diagnostics import saved_recovery_error

    handler, _, _ = recovery_handler(
        tmp_path, monkeypatch, {"kind": "answer", "answer": "只读解释"}
    )
    handler([{"role": "user", "content": "解释回收问题"}], conversation_id="c")
    assert saved_recovery_error(handler.state_path)
    monkeypatch.setattr(recovery, "recover_saved_results", lambda current, **kw: current)
    handler([{"role": "user", "content": "解释回收问题"}], conversation_id="c")
    assert saved_recovery_error(handler.state_path)
    monkeypatch.setattr(
        recovery, "recovery_candidate", lambda *a, **kw: {"pending": True, "fingerprint": "return2"}
    )
    from phase_agent.graphs.dialogue import saved_control

    monkeypatch.setattr(saved_control, "semantic_recovery_intent", lambda *a, **kw: "approve")

    def success(current, **kw):
        return {**deepcopy(current), "result_collection": {"recovered_count": 1}}

    monkeypatch.setattr(recovery, "recover_saved_results", success)
    # Stop only the post-success scientific decision in this diagnostic-only test.
    handler._run = lambda *a, **kw: {
        "status": "answered",
        "answer": "回收成功",
        "state": read_json(handler.state_path, {}),
    }
    handler([{"role": "user", "content": "已回传"}], conversation_id="c")
    assert saved_recovery_error(handler.state_path) is None
    reply = handler([{"role": "user", "content": "当前状态"}], conversation_id="c")
    assert "结果回收失败" not in reply


@pytest.mark.parametrize(
    "content", ["{broken", "[]", '{"state_path":"old-path","error":"old failure"}']
)
def test_invalid_diagnostic_is_visible_but_cannot_block_legitimate_retry(
    tmp_path, monkeypatch, content
):
    from pathlib import Path
    from phase_agent.tools.workflows import recover_saved_results as recovery
    from phase_agent.runtime.recovery_diagnostics import saved_recovery_error

    handler = make_handler(tmp_path, lambda p: {"kind": "answer", "answer": "当前事实"})
    write_json(handler.state_path, {})
    Path(str(handler.state_path) + ".recovery-error.json").write_text(content, encoding="utf-8")
    assert saved_recovery_error(handler.state_path) is None
    reply = handler([{"role": "user", "content": "当前状态"}], conversation_id="c")
    assert "诊断记录" in reply
    reply = handler([{"role": "user", "content": "说明当前情况"}], conversation_id="c")
    assert "当前事实" in reply


@pytest.mark.parametrize(
    "report",
    [
        {"recovered_count": 0, "invalid_count": 1},
        {"recovered_count": 0, "missing_marker_count": 1},
        {"recovered_count": 0, "missing_structure_count": 1},
        {"recovered_count": 0, "transfer_errors": ["synthetic missing archive"]},
    ],
)
def test_zero_verified_results_with_failed_checks_never_acknowledges(tmp_path, monkeypatch, report):
    from phase_agent.tools.workflows import recover_saved_results as recovery
    from phase_agent.graphs.dialogue import saved_control

    actual_recover = recovery.recover_saved_results
    handler, _, state = recovery_handler(
        tmp_path, monkeypatch, {"kind": "answer", "answer": "回收核验错误"}
    )
    state["pending_execution_policies"] = {"old-plan": {"agent_proposal": {}}}
    write_json(handler.state_path, state)
    monkeypatch.setattr(
        recovery,
        "recovery_candidate",
        lambda *a, **kw: {"pending": True, "fingerprint": "invalid-return"},
    )
    monkeypatch.setattr(saved_control, "semantic_recovery_intent", lambda *a, **kw: "approve")
    monkeypatch.setattr(recovery, "recover_saved_results", actual_recover)

    class Collector:
        def collect_results_with_report(self, current):
            return {"results": [], "report": deepcopy(report)}

    handler.workflow_kwargs["result_collector"] = Collector()
    reply = handler([{"role": "user", "content": "已回传"}], conversation_id="c")
    assert "没有通过本次回收核验" in reply
    assert read_json(handler.state_path, {}) == state


def test_task_diagnostics_never_override_training_or_old_metadata(tmp_path):
    from phase_agent.tools.workflows.recover_saved_results import _empty_collection_failure

    assert _empty_collection_failure({"missing_result_count": 2}, {}) is None
    assert _empty_collection_failure({"metadata_refresh_report": {"invalid_count": 1}}, {}) is None
    results = tmp_path / "training/results"
    results.mkdir(parents=True)
    (results / "training.finished.json").write_text("{}")
    state = {
        "active_model_version": "base",
        "remote_finetune_jobs": {
            "j": {
                "directory": str(results.parent),
                "status": "waiting_results",
                "original_model_version": "base",
            }
        },
    }
    assert _empty_collection_failure({"invalid_count": 1}, state) is None
    state["remote_finetune_jobs"]["j"]["activated"] = True
    assert _empty_collection_failure({"invalid_count": 1}, state)

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from phase_agent.runtime.chat_execution import run_workflow_turn


@pytest.mark.parametrize("unsettled,returned", [(False, False), (True, False), (False, True)])
def test_stale_checkpoint_preserves_registered_training_wait_only_when_receipts_clear(
    tmp_path, monkeypatch, unsettled, returned
):
    from tests.test_training_delivery_receipt import prepared_state
    import phase_agent.tools.state.execution_receipts as receipts

    state = prepared_state(tmp_path, "submitted")
    if returned:
        output = tmp_path / "Finetune_Batch01_output"
        output.mkdir()
        (output / "training.results.zip").write_bytes(b"new output requires recovery")
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    before = path.read_bytes()
    report = {"status": "clear", "unsettled": []}
    if unsettled:
        report = {"status": "reconciliation_required", "unsettled": [{"reason": "real_execution"}]}
    monkeypatch.setattr(receipts, "recovery_report", lambda *_: report)
    calls = []

    def workflow(**kwargs):
        calls.append(kwargs)
        raise ValueError("Cannot replay a new invocation after an execution or approval stage")

    handler = SimpleNamespace(
        state_path=path, workflow_kwargs={}, workflow=workflow, unified_dialogue=False
    )
    result = run_workflow_turn(handler, "another-message", None, "继续")
    assert result["status"] == (
        "execution_reconciliation_required" if unsettled or returned else "awaiting_remote_training"
    )
    assert result["steps_executed"] == 0 and result["submitted"] is False
    assert len(calls) == 1 and path.read_bytes() == before
    if not unsettled and not returned:
        from phase_agent.runtime.workflow_reply_presentation import format_workflow_reply

        reply = format_workflow_reply(result, path)
        assert "等待训练结果" in reply and "旧动作" not in reply
        assert "Finetune_Batch01_output" in reply


@pytest.mark.parametrize(
    "message",
    [
        "Cannot replay a new invocation after an execution or approval stage",
        "Lifecycle configuration changed; cannot replay old invocation",
    ],
)
def test_stale_lifecycle_error_becomes_read_only_reconciliation(tmp_path, message):
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"tasks": [{"task_id": "t1"}]}), encoding="utf-8")
    before = state_path.read_bytes()
    calls = []

    def workflow(**kwargs):
        calls.append(kwargs)
        raise ValueError(message)

    handler = SimpleNamespace(
        workflow_kwargs={"state_path": str(state_path)},
        state_path=Path(state_path),
        workflow=workflow,
        unified_dialogue=False,
    )
    result = run_workflow_turn(handler, "new-request", None, "继续")

    assert result["status"] == "execution_reconciliation_required"
    assert result["submitted"] is False
    assert result["recovery_report"]["unsettled"][0]["identity"]["invocation_id"] == "new-request"
    assert result["recovery_report"]["unsettled"][0]["reason"] == "stale_lifecycle_checkpoint"
    assert len(calls) == 1
    assert state_path.read_bytes() == before

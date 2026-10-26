import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from phase_agent.runtime.chat_execution import run_workflow_turn


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

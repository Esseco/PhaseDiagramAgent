import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from phase_agent.runtime.local_agent_control import LocalAgentControl


def test_report_existing_plan_preserves_state_and_writes_agent_markdown(tmp_path):
    state_path = tmp_path / "workflow_state" / "state.json"
    state_path.parent.mkdir()
    state = {
        "confirmed_config": {"storage": {"workspace_root": str(tmp_path)}},
        "pending_execution_policies": {
            "plan-1": {"agent_proposal": {"recommended_action": "generate_branches"}}
        },
    }
    original = json.dumps(state)
    state_path.write_text(original, encoding="utf-8")
    handler = SimpleNamespace(state_path=state_path, lock=threading.RLock())
    control = LocalAgentControl(handler)
    with patch(
        "phase_agent.runtime.workflow_reply_presentation.format_workflow_reply",
        return_value="待审批：生成 branch；仅登记，不计算。",
    ):
        result = control.codex_report("plan-1")
    report = Path(result["report_path"])
    assert report.is_file()
    assert "待审批" in report.read_text(encoding="utf-8")
    assert state_path.read_text(encoding="utf-8") == original


def test_report_requires_existing_pending_plan(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{}", encoding="utf-8")
    control = LocalAgentControl(SimpleNamespace(state_path=path, lock=threading.RLock()))
    with pytest.raises(ValueError, match="pending_plan_not_found"):
        control.codex_report("missing")
    assert list(tmp_path.iterdir()) == [path]


def test_report_rejects_state_outside_project(tmp_path):
    from phase_agent.runtime.codex.reports import save_codex_report

    with pytest.raises(ValueError, match="report_state_outside_project_workspace"):
        save_codex_report(tmp_path / "other" / "state.json", {
            "state": {"workspace_layout": {"workspace_root": str(tmp_path / "project")}}
        })


def test_proposal_report_failure_preserves_approval_response(tmp_path):
    path = tmp_path / "state.json"
    handler = SimpleNamespace(
        state_path=path,
        lock=threading.RLock(),
        workflow_kwargs={},
        _run=lambda *args: {"status": "awaiting_approval", "record_id": "p1"},
    )
    control = LocalAgentControl(handler)
    snapshot = {
        "context_version": "v1", "context_digest": "d1", "allowed_tools": ["pause_search"],
        "decision_context": {}, "state": {},
    }
    with patch("phase_agent.runtime.codex.context.build_codex_context", return_value=snapshot), patch(
        "phase_agent.runtime.codex.reports.save_codex_report", side_effect=OSError("disk error")
    ):
        response = control.propose_codex(
            {"tool": "pause_search", "parameters": {}, "reason": "审阅"},
            context_version="v1", context_digest="d1",
        )
    assert response["reply"]["status"] == "awaiting_approval"
    assert "勿重复提交" in response["report_error"]
    assert handler.workflow_kwargs == {}

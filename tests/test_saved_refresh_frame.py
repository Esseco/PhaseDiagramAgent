from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
import pytest

from phase_agent.tools.workflows.saved_refresh_frame import prepare_saved_refresh_frame


def frame():
    config = {"system": {"name": "Ni"}}
    current = {
        "confirmed_config_version": "C", "confirmed_config": config,
        "active_model_version": "M2", "active_model": {"version": "M2"},
        "model_refresh": {"status": "approval_required", "old_model_version": "M1",
                          "new_model_version": "M2", "old_diagram": {"status": "completed", "model_version": "M1"}},
        "pending_execution_policies": {}, "tasks": [],
    }
    return {"loaded_state": current, "snapshot": {"config_version": "C", "config": deepcopy(config)},
            "execution_mode": "interactive", "state_path": "state.json", "effective_config": {},
            "agent_client": SimpleNamespace(_codex_structured=True, _frozen_refresh_draft=True)}


def execute(f, *, active=False, returned=False, candidate=None, receipt=None):
    with (patch("phase_agent.policies.workflow_policy.active_tasks", return_value=[1] if active else []),
          patch("phase_agent.tools.remote.transfer_bundle.has_registered_return", return_value=returned),
          patch("phase_agent.tools.workflows.recover_saved_results.recovery_candidate", return_value=candidate or {"pending": False, "backlog_count": 0}),
          patch("phase_agent.tools.state.execution_receipts.recovery_report", return_value=receipt or {"status": "clear", "unsettled": []})):
        return prepare_saved_refresh_frame(f)


def test_idle_refresh_preview_and_frozen_approval_reuse_existing_gate():
    f = frame()
    result = execute(f)
    assert result["saved_input_only"] is True
    assert result["pre_reconciled"]["state"] is f["loaded_state"]
    assert result["feedback"]["recorded"] == []
    action = {"tool": "prepare_local_batch_files", "parameters": {"mode": "model_refresh_inputs"}}
    f["loaded_state"]["pending_execution_policies"] = {"P": {"config_version": "C", "agent_proposal": {"raw_action": action}}}
    f.update(invocation_id="P", human_feedback={"decision": "approve"})
    assert execute(f)["saved_input_only"] is True


@pytest.mark.parametrize("change", ["config", "model", "diagram", "active", "returned", "backlog", "unsettled", "training", "other_pending", "error", "recovered"])
def test_fast_path_fails_closed_when_evidence_or_work_changes(change):
    f = frame(); state = f["loaded_state"]; kwargs = {}
    if change == "config": f["snapshot"]["config_version"] = "changed"
    if change == "model": state["active_model_version"] = "other"
    if change == "diagram": state["model_refresh"]["old_diagram"]["status"] = "invalidated"
    if change == "active": kwargs["active"] = True
    if change == "returned": kwargs["returned"] = True
    if change == "backlog": kwargs["candidate"] = {"pending": True, "backlog_count": 1}
    if change == "unsettled": kwargs["receipt"] = {"status": "reconciliation_required", "unsettled": [{"reason": "inflight"}]}
    if change == "training": state["remote_finetune_jobs"] = {"J": {"status": "inputs_prepared"}}
    if change == "other_pending": state["pending_execution_policies"] = {"other": {}}
    if change == "error": state["result_collection"] = {"recovery_error": "unresolved"}
    if change == "recovered": f["recovered_results"] = [{"task_id": "T"}]
    assert execute(f, **kwargs) is f

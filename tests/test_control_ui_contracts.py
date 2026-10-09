import json
from types import SimpleNamespace
import pytest
from run.local_agent_control import LocalAgentControl
from run.control_ui_contracts import PendingResponse, StatusResponse, validate_control_response


def test_read_only_endpoints_preserve_hash_and_state(tmp_path):
    path = tmp_path / "state.json"
    state = {"pending_execution_policies": {"p1": {"revision": 2,
        "agent_proposal": {"recommended_action": "update_mlip", "action_parameters": {},
        "raw_action": {"tool": "update_mlip", "target_ids": []}, "reason": "待批准"}}}}
    path.write_text(json.dumps(state), encoding="utf-8")
    before = path.read_bytes()
    control = LocalAgentControl(SimpleNamespace(state_path=path))
    status = control.status()
    pending = control.pending()
    assert status["pending_approvals"] == pending["count"] == 1
    assert pending["pending"][0]["proposal_hash"]
    assert pending["state_version"] == status["summary_id"]
    assert path.read_bytes() == before
    assert control.pending() == pending


@pytest.mark.parametrize("change", [{"count": True}, {"count": 1}, {"state_version": []}])
def test_pending_malformed_output_fails_without_echoing_values(change):
    response = {"pending": [], "count": 0, "state_version": "s1", "approval_url": "local", **change}
    with pytest.raises(ValueError, match="Invalid UI response"):
        validate_control_response(PendingResponse, response)


def test_negative_counts_and_nonfinite_cost_are_rejected(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{}")
    control = LocalAgentControl(SimpleNamespace(state_path=path))
    response = control.status()
    response["reserved_relative_cost"] = float("nan")
    with pytest.raises(ValueError):
        validate_control_response(StatusResponse, response)
    response["reserved_relative_cost"] = 0
    response["task_counts"] = {"pending": -1}
    with pytest.raises(ValueError):
        validate_control_response(StatusResponse, response)


def test_tasks_config_memory_response_preserves_state(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"tasks": [{"task_id": "T1", "stage": "deep_search"}],
        "decision_memory": {"records": [{"knowledge_id": "K1"}]}}))
    before = path.read_bytes()
    handler = SimpleNamespace(state_path=path, workflow_kwargs={"config_session": {
        "status": "draft", "draft_revision": 1, "config": {"name": "test"}}})
    control = LocalAgentControl(handler)
    assert control.tasks()["tasks"][0]["task_id"] == "T1"
    assert control.config()["config"] == {"name": "test"}
    assert control.memory()["records"] == [{"knowledge_id": "K1"}]
    assert path.read_bytes() == before

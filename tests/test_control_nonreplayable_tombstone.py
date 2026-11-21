import hashlib
import json
import pytest
from phase_agent.tools.state.workflow_control import commit_control, control_path


def test_nonreplayable_history_does_not_block_new_control_and_keeps_guard(tmp_path):
    path = tmp_path / "state.json"
    state = {"workflow_control_revision":"old","tasks":[]}
    path.write_text(json.dumps(state))
    control = {"state_path":str(path.resolve()),"revision":"old","restore_allowed":False,"binding":{},"patch":{"cancelled_waits":{"missing":{"task_id":"missing"}}}}
    control_path(path).write_text(json.dumps(control))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    updated = commit_control(path,state,{},evidence={"reason":"test"},expected_sha256=digest,desired_state=state,restore_allowed=False)
    stored = json.loads(control_path(path).read_text())
    assert stored["patch"]["cancelled_waits"]["missing"]["task_id"] == "missing"
    assert stored["restore_allowed"] is False
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match="cancelled_task_wait_binding_changed"):
        commit_control(path,updated,{"cancelled_waits":{"new":{"task_id":"new"}}},evidence={"reason":"test"},expected_sha256=digest,desired_state=updated,restore_allowed=False)

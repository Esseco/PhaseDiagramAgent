import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest


def fixture(tmp_path, monkeypatch):
    from phase_agent.analysis.convergence import independent_evidence

    monkeypatch.setattr(independent_evidence, "geometry_key", lambda row: __import__("pathlib").Path(row["structure"]).read_text())
    old = tmp_path / "old.vasp"
    new = tmp_path / "new.vasp"
    old.write_text("old")
    new.write_text("new")
    old_sha = hashlib.sha256(old.read_bytes()).hexdigest()
    new_sha = hashlib.sha256(new.read_bytes()).hexdigest()
    task_input = tmp_path / "task.json"
    task_input.write_text(json.dumps({"task_id": "old", "task_checksum": "t", "input_structure_sha256": old_sha, "structure_path": str(old)}))
    task = {"task_id": "old", "stage": "dft_single_point", "status": "pending", "batch_id": "b", "input_path": str(task_input), "task_checksum": "t", "input_structure_sha256": old_sha}
    candidate = {"candidate_id": "new", "structure_path": str(new), "structure_sha256": new_sha, "model_version": "m", "qbc_model_version": "m", "qbc_frame_sha256": new_sha, "qbc": {"status": "completed"}}
    action = {"tool": "select_dft_candidates", "operation": "select_dft_candidates", "target_ids": ["new"], "parameters": {"decisions": [{"candidate_id": "new", "action": "DFT_SINGLE_POINT"}], "prepare_inputs_after_approval": True}, "_decision_team": {"backend": "codex_native"}, "_simulation_plan": {"action_hash": "frozen"}}
    state = {"active_model_version": "m", "tasks": [task], "qbc_candidates": [candidate], "budget_usage": {"stages": {"dft_single_point": {"tasks": 1}}}, "budget_reservations": {"old": {"stage": "dft_single_point", "status": "reserved"}}}
    wait = {"task_ids": ["old"], "waiting_task_count": 1, "waiting_by_stage": {"dft_single_point": 1}}
    config = {"budgets": {"stage_limits": {"dft_single_point": {"max_tasks": 3}}}}
    frame = {"feedback": {"state": state}, "manual_wait": wait, "effective_config": config, "agent_client": SimpleNamespace(_frozen_dft_action=action), "human_feedback": None}
    return frame, action, new


def test_frozen_disjoint_dft_can_enter_approval_without_waiving_old(tmp_path, monkeypatch):
    from phase_agent.tools.workflows.manual_dft_coexistence import allowed
    frame, _, _ = fixture(tmp_path, monkeypatch)
    before = deepcopy(frame["feedback"]["state"])
    assert allowed(frame)
    assert frame["feedback"]["state"] == before


@pytest.mark.parametrize("bad", ["same_geometry", "wrong_hash", "bad_qbc", "other_stage", "running", "limit", "not_frozen"])
def test_dft_coexistence_fails_closed(tmp_path, monkeypatch, bad):
    from phase_agent.tools.workflows.manual_dft_coexistence import allowed
    frame, action, new = fixture(tmp_path, monkeypatch)
    state = frame["feedback"]["state"]
    if bad == "same_geometry":
        new.write_text("old")
        digest = hashlib.sha256(new.read_bytes()).hexdigest()
        state["qbc_candidates"][0].update(structure_sha256=digest, qbc_frame_sha256=digest)
    elif bad == "wrong_hash":
        state["qbc_candidates"][0]["structure_sha256"] = "invalid"
    elif bad == "bad_qbc":
        state["qbc_candidates"][0]["qbc_model_version"] = "old"
    elif bad == "other_stage":
        frame["manual_wait"]["waiting_by_stage"] = {"deep_search": 1}
    elif bad == "running":
        state["tasks"][0]["status"] = "running"
    elif bad == "limit":
        frame["effective_config"]["dft"] = {"selection": {"max_tasks_per_round": 0}}
    else:
        frame["agent_client"] = None
    assert not allowed(frame)


def test_approval_resumes_only_its_matching_frozen_plan(tmp_path, monkeypatch):
    from phase_agent.tools.workflows.manual_dft_coexistence import allowed
    frame, action, _ = fixture(tmp_path, monkeypatch)
    frame["agent_client"] = None
    frame["invocation_id"] = "plan1"
    frame["human_feedback"] = {"decision": "approve"}
    frame["feedback"]["state"]["pending_execution_policies"] = {"plan1": {"agent_proposal": {"raw_action": action}}}
    assert allowed(frame)
    frame["invocation_id"] = "other"
    assert not allowed(frame)
    frame["invocation_id"] = "plan1"
    frame["human_feedback"] = {"decision": "reject"}
    assert not allowed(frame)

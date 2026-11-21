"""Administrative DFT decisions must not overwrite concurrent project updates."""

import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from phase_agent.runtime.local_agent_control import LocalAgentControl
from phase_agent.tools.step_runner.file_protocol import read_json, write_json
from tests.test_dft_failure_policy import make_state


def _concurrent_update(path):
    script = (
        "import sys; from phase_agent.tools.step_runner.file_protocol import read_json, write_json; "
        "p=sys.argv[1]; s=read_json(p); s['other_process_update']=True; write_json(p,s)"
    )
    subprocess.run([sys.executable, "-c", script, str(path)], check=True)


def test_failure_diagnosis_rejects_concurrent_change(tmp_path, monkeypatch):
    import phase_agent.evidence.dft_failure_policy as policy

    state, _ = make_state(2, 8)
    state["active_model_version"] = "m"
    path = tmp_path / "state.json"
    write_json(path, state)
    batch = policy.completed_failure_batches(state, [])[0]
    original = policy.completed_failure_batches

    def concurrent(current, rows):
        result = original(current, rows)
        _concurrent_update(path)
        return result

    monkeypatch.setattr(policy, "completed_failure_batches", concurrent)
    control = LocalAgentControl(SimpleNamespace(state_path=path, lock=threading.RLock()))
    diagnosis = {
        "assessment": "inspect input",
        "evidence": ["execution.operation_outcome"],
        "recommendation": "return_to_strategic",
        "limitations": "logs unavailable",
    }
    with pytest.raises(ValueError, match="stale_state_file_version"):
        control.register_dft_failure_analysis(batch["failure_evidence_digest"], diagnosis)
    saved = read_json(path)
    assert saved["other_process_update"] is True
    assert not saved.get("dft_failure_reviews")


def test_wait_waiver_rejects_concurrent_change(tmp_path, monkeypatch):
    import phase_agent.tools.state.dft_recovery_decision as decision

    state, _ = make_state(2, 2, pending=2)
    for task in state["tasks"]:
        task["search_group_index"] = 0
    path = tmp_path / "state.json"
    write_json(path, state)
    original = decision.update_dft_recovery_question

    def concurrent(current, *args, **kwargs):
        result = original(current, *args, **kwargs)
        if args:
            _concurrent_update(path)
        return result

    monkeypatch.setattr(decision, "update_dft_recovery_question", concurrent)
    control = LocalAgentControl(SimpleNamespace(state_path=path, lock=threading.RLock()))
    with pytest.raises(ValueError, match="stale_state_file_version"):
        control.register_dft_wait_waiver(["4", "5"], "不再等待这两项")
    saved = read_json(path)
    assert saved["other_process_update"] is True
    assert all(task["status"] == "pending" for task in saved["tasks"][4:])
    assert not saved.get("dft_recovery_cancellations")


def test_candidate_refresh_rejects_concurrent_change(tmp_path, monkeypatch):
    import phase_agent.science.qbc.post_mc_candidates as module

    path = tmp_path / "state.json"
    state = {
        "active_model_version": "m",
        "phase_diagrams": {"mlip": {"status": "completed", "model_version": "m", "version": "h"}},
    }
    write_json(path, state)

    def concurrent(current, model):
        _concurrent_update(path)
        return [{"candidate_id": "new"}]

    monkeypatch.setattr(module, "post_mc_candidates", concurrent)
    control = LocalAgentControl(SimpleNamespace(state_path=path, lock=threading.RLock()))
    with pytest.raises(ValueError, match="stale_state_file_version"):
        control.refresh_dft_candidates()
    saved = read_json(path)
    assert saved["other_process_update"] is True
    assert "qbc_candidates" not in saved


def test_direct_binding_rejects_concurrent_change(tmp_path, monkeypatch):
    import phase_agent.tools.workflows.direct_review_report as module

    path = tmp_path / "state.json"
    write_json(path, {"active_model_version": "m"})

    def concurrent(current, ledger, scope):
        _concurrent_update(path)
        return {**current, "qbc_candidates": [{"candidate_id": "new"}]}, {"status": "bound"}

    monkeypatch.setattr(module, "bind_direct_candidates", concurrent)
    handler = SimpleNamespace(
        state_path=path,
        lock=threading.RLock(),
        workflow_kwargs={"manager": SimpleNamespace(data={})},
    )
    control = LocalAgentControl(handler)
    with pytest.raises(ValueError, match="stale_state_file_version"):
        control.bind_direct_candidates("scope")
    saved = read_json(path)
    assert saved["other_process_update"] is True
    assert "qbc_candidates" not in saved

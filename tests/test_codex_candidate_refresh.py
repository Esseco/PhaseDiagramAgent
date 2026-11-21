import json
import threading
from types import SimpleNamespace

import pytest

from phase_agent.runtime.local_agent_control import LocalAgentControl


def test_refresh_candidates_preserves_state_and_rejects_pending(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    state = {
        "active_model_version": "m1",
        "phase_diagrams": {"mlip": {"status": "completed", "model_version": "m1", "version": "h1"}},
        "qbc_candidates": [{"candidate_id": "old"}],
        "tasks": [{"task_id": "complete", "status": "completed"}],
    }
    path.write_text(json.dumps(state), encoding="utf8")
    control = LocalAgentControl(SimpleNamespace(state_path=path, lock=threading.RLock()))
    monkeypatch.setattr(
        "phase_agent.science.qbc.post_mc_candidates.post_mc_candidates",
        lambda state, version: [{"candidate_id": "new"}],
    )
    result = control.refresh_dft_candidates()
    saved = json.loads(path.read_text(encoding="utf8"))
    assert result["candidate_count"] == 1
    assert saved["qbc_candidates"] == [{"candidate_id": "new"}]
    assert saved["tasks"] == state["tasks"]
    saved["pending_execution_policies"] = {"p": {"status": "awaiting_approval"}}
    path.write_text(json.dumps(saved), encoding="utf8")
    with pytest.raises(ValueError, match="pending"):
        control.refresh_dft_candidates()
    assert json.loads(path.read_text(encoding="utf8")) == saved


def test_refresh_receipt_and_candidates_use_same_latest_hull(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    total = {"status": "completed", "model_version": "m", "version": "total",
             "energy_policy": "latest_valid_per_structure_v1", "mixed_model_versions": True, "entries": []}
    independent = {"status": "completed", "model_version": "m", "version": "independent", "entries": []}
    state = {"active_model_version": "m", "phase_diagrams": {"mlip": total},
             "phase_diagrams_by_model": {"m": independent}}
    path.write_text(json.dumps(state), encoding="utf8")
    control = LocalAgentControl(SimpleNamespace(state_path=path, lock=threading.RLock()))
    monkeypatch.setattr("phase_agent.science.qbc.post_mc_candidates.post_mc_candidates",
                        lambda state, version: [{"candidate_id": "c", "phase_diagram_version": "total"}])
    result = control.refresh_dft_candidates()
    saved = json.loads(path.read_text(encoding="utf8"))
    assert result["hull_version"] == "total"
    assert saved["qbc_candidates"][0]["phase_diagram_version"] == result["hull_version"]

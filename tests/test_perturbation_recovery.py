from hashlib import sha256
from copy import deepcopy
import pytest
from phase_agent.evidence.perturbation_recovery import build_perturbation_recovery_pool


def test_pool_requires_normal_verified_frame_and_small_h(tmp_path):
    frame = tmp_path / "frame.vasp"
    frame.write_bytes(b"frame")
    digest = sha256(frame.read_bytes()).hexdigest()
    entry = {"record_id": "r", "structure_id": "s", "structure_sha256": digest,
             "structure_path": str(frame), "phase_identification_status": "identified",
             "phase": "O3", "x_Na_per_O2": 0.5}
    snapshot = {"status": "completed", "version": "h", "model_version": "old", "entries": [entry]}
    task = {"task_id": "t", "status": "completed", "branch_id": "b",
            "outputs": {"structure_checksum": digest, "extrapolation_monitor": {"extrapolation_failure": False, "failures": [], "final_feature_difference": 0.4}}}
    state = {"phase_diagrams": {"mlip": snapshot}, "phase_records": [{"record_id": "r", "structure_id": "s", "structure_sha256": digest, "source_task_id": "t"}], "tasks": [task]}
    ledger = {"branches": {"b": {"H": [[2,0,0],[0,1,0],[0,0,1]]}}}
    result = build_perturbation_recovery_pool(state, ledger, max_det_H=2)
    assert len(result["candidates"]) == 1 and result["candidates"][0]["det_H"] == 2
    assert build_perturbation_recovery_pool(state, ledger, max_det_H=1)["candidates"] == []
    task["outputs"]["extrapolation_monitor"]["extrapolation_failure"] = True
    assert build_perturbation_recovery_pool(state, ledger, max_det_H=2)["candidates"] == []
    task["outputs"]["extrapolation_monitor"]["extrapolation_failure"] = False
    frame.write_bytes(b"tampered")
    assert build_perturbation_recovery_pool(state, ledger, max_det_H=2)["candidates"] == []


def test_pool_rejects_unverified_hull_or_det():
    with pytest.raises(ValueError, match="valid_hull"):
        build_perturbation_recovery_pool({}, {}, max_det_H=2)
    with pytest.raises(ValueError, match="max_det_H"):
        build_perturbation_recovery_pool({}, {}, max_det_H=7)


def test_historical_endpoint_evidence_binds_input_hash(tmp_path, monkeypatch):
    import json
    from phase_agent.evidence.perturbation_recovery import _normal_frame_digests
    from phase_agent.evidence import extrapolation_review as review_module
    csv_path = tmp_path / "evidence.csv"
    csv_path.write_text("task_id,score,classification,error\nt1,0.9,below_threshold,\nt2,1.1,extrapolation_failure,\n", encoding="utf8")
    state = {"active_model_version": "bad", "extrapolation_review_summary": {"screened_count": 2}, "tasks": [{"task_id": "t1", "model_version": "bad", "input_structure_sha256": "good"}, {"task_id": "t2", "model_version": "bad", "input_structure_sha256": "bad", "extrapolation_review": {"status": "extrapolation_failure", "source": str(csv_path)}}]}
    monkeypatch.setattr(review_module, "extrapolation_review", lambda state: {"status": "endpoint_classified_incomplete"})
    result = _normal_frame_digests(state)
    assert result == {"good": "initial_final_feature_difference"}
    monkeypatch.setattr(review_module, "extrapolation_review", lambda state: {"status": "unverified"})
    with pytest.raises(ValueError, match="endpoint_evidence_invalid"):
        _normal_frame_digests(state)


def test_unimplemented_nepkit_contract_does_not_claim_prepared():
    from phase_agent.actions.nepkit_initialization import prepare_nepkit_initial_dataset
    result = prepare_nepkit_initial_dataset(action={"parameters": {"initial_structures": ["frame"], "boundary": {}, "confirmed": True}})
    assert result["status"] == "not_configured" and result["submitted"] is False


def test_conflicting_failure_excludes_same_input_frame(tmp_path, monkeypatch):
    from phase_agent.evidence.perturbation_recovery import _normal_frame_digests
    from phase_agent.evidence import extrapolation_review as review_module
    csv_path = tmp_path / "rows.csv"
    csv_path.write_text("task_id,score,classification,error\nt1,0.2,below_threshold,\nt2,1.2,extrapolation_failure,\n", encoding="utf8")
    state = {"active_model_version": "m", "extrapolation_review_summary": {}, "tasks": [{"task_id": "t1", "model_version": "m", "input_structure_sha256": "same"}, {"task_id": "t2", "model_version": "m", "input_structure_sha256": "same", "extrapolation_review": {"status": "extrapolation_failure", "source": str(csv_path)}}]}
    state["extrapolation_review_summary"] = {"screened_count": 2}
    monkeypatch.setattr(review_module, "extrapolation_review", lambda state: {"status": "endpoint_classified_incomplete"})
    assert _normal_frame_digests(state) == {}


def test_pool_uses_pre_refresh_hull_during_failed_round(monkeypatch):
    from phase_agent.agents.reliability import extrapolation_round
    monkeypatch.setattr(extrapolation_round, 'required_direct_round', lambda state: {'scope_id': 'epoch_refresh'})
    old = {'status': 'completed', 'version': 'old-hull', 'model_version': 'old', 'entries': []}
    failed = {'status': 'completed', 'version': 'bad-hull', 'model_version': 'bad', 'entries': []}
    state = {'model_refresh': {'old_diagram': old, 'old_model_version': 'old'}, 'phase_diagrams': {'mlip': failed}}
    result = build_perturbation_recovery_pool(state, {}, max_det_H=2)
    assert result['hull_version'] == 'old-hull'
    state['model_refresh']['old_model_version'] = 'mismatch'
    with pytest.raises(ValueError, match='hull_binding_invalid'):
        build_perturbation_recovery_pool(state, {}, max_det_H=2)

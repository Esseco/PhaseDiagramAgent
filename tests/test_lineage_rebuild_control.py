from threading import RLock
from types import SimpleNamespace

import pytest

from phase_agent.runtime.local_agent_control import LocalAgentControl
from phase_agent.tools.step_runner.file_protocol import read_json, write_json


def test_formal_lineage_rebuild_preserves_history_and_requires_evidence(tmp_path, monkeypatch):
    from phase_agent.analysis.phase import update_phase_diagram as module

    path = tmp_path / "state.json"
    state = {"active_model_version": "m", "phase_records": [{"record_id": "r"}],
             "model_refresh": {"new_model_version": "m", "status": "waiting_results"},
             "phase_diagrams": {"mlip": {"version": "old", "entries": [{"record_id": "old"}]},
                                "dft": {"version": "dft"}}}
    write_json(path, state)
    control = LocalAgentControl(SimpleNamespace(state_path=path, lock=RLock()))
    result = {"diagrams": {"mlip": {"version": "new", "model_version": "m", "status": "completed",
                                     "entries": [{"record_id": "r"}]}}}
    monkeypatch.setattr(module, "update_phase_diagram", lambda *a, **k: result)
    with pytest.raises(ValueError, match="missing_required"):
        control.rebuild_mlip_lineage(expected_model="m", expected_previous_version="old",
                                     required_record_ids=["missing"])
    assert read_json(path, {}) == state
    report = control.rebuild_mlip_lineage(expected_model="m", expected_previous_version="old",
                                          required_record_ids=["r"])
    saved = read_json(path, {})
    assert report["status"] == "rebuilt" and not report["submitted"]
    assert saved["phase_diagrams"]["dft"] == {"version": "dft"}
    assert saved["phase_records"] == state["phase_records"]
    assert saved["lineage_rebuild_history"][0]["previous_diagram"] == state["phase_diagrams"]["mlip"]
    with pytest.raises(ValueError, match="context_changed"):
        control.rebuild_mlip_lineage(expected_model="m", expected_previous_version="old",
                                     required_record_ids=["r"])


def test_completed_outside_rebuild_requires_exact_removed_evidence(tmp_path, monkeypatch):
    from phase_agent.analysis.phase import update_phase_diagram as module
    from phase_agent.analysis.phase import outside_refresh_hull

    path = tmp_path / "state.json"
    old = {"record_id": "old", "structure_id": "s", "structure_sha256": "sha", "source_version": "old-model"}
    keep = {"record_id": "keep", "structure_id": "k", "structure_sha256": "ksha", "source_version": "m"}
    state = {"active_model_version": "m", "phase_records": [], "tasks": [],
             "upload_layout": {"model_rounds": {"old-model": 1, "m": 2}},
             "model_refresh": {"new_model_version": "m", "status": "completed"},
             "phase_diagrams": {"mlip": {"version": "old", "entries": [old, keep]}}}
    write_json(path, state)
    control = LocalAgentControl(SimpleNamespace(state_path=path, lock=RLock()))
    result = {"diagrams": {"mlip": {"version": "new", "model_version": "m", "status": "completed", "entries": [keep]}}}
    monkeypatch.setattr(module, "update_phase_diagram", lambda *a, **k: result)
    monkeypatch.setattr(outside_refresh_hull, "outside_refresh_inputs", lambda *a, **k: {("s", "sha"): 2})
    kwargs = dict(expected_model="m", expected_previous_version="old",
                  reason="remove superseded outside-boundary frames")
    with pytest.raises(ValueError, match="outside_scope_changed"):
        control.rebuild_mlip_lineage(required_record_ids=["keep"], **kwargs)
    assert read_json(path, {}) == state
    monkeypatch.setattr(outside_refresh_hull, "outside_refresh_inputs", lambda *a, **k: {})
    with pytest.raises(ValueError, match="outside_evidence_missing"):
        control.rebuild_mlip_lineage(required_record_ids=["old"], **kwargs)
    assert read_json(path, {}) == state
    monkeypatch.setattr(outside_refresh_hull, "outside_refresh_inputs", lambda *a, **k: {("s", "sha"): 2})
    report = control.rebuild_mlip_lineage(required_record_ids=["old"], **kwargs)
    saved = read_json(path, {})
    assert report["status"] == "rebuilt" and saved["phase_diagrams"]["mlip"]["version"] == "new"
    assert saved["lineage_rebuild_history"][-1]["previous_diagram"] == state["phase_diagrams"]["mlip"]
    assert saved["model_refresh"] == state["model_refresh"]


def test_unidentified_input_rebuild_requires_exact_bound_frame(tmp_path, monkeypatch):
    from phase_agent.analysis.phase import update_phase_diagram as module, outside_refresh_hull
    path = tmp_path / "state.json"
    old = {"record_id":"old", "structure_id":"s", "structure_sha256":"sha", "source_version":"m0"}
    keep = {"record_id":"keep", "structure_id":"k", "structure_sha256":"ksha", "source_version":"m"}
    state = {"active_model_version":"m", "phase_records":[], "tasks":[],
             "upload_layout":{"model_rounds":{"m0":1,"m":2}},
             "model_refresh":{"new_model_version":"m","status":"completed"},
             "phase_diagrams":{"mlip":{"version":"old","entries":[old,keep]}}}
    write_json(path,state)
    control = LocalAgentControl(SimpleNamespace(state_path=path,lock=RLock()))
    monkeypatch.setattr(module,"update_phase_diagram",lambda *a,**k:{"diagrams":{"mlip":{"version":"new","model_version":"m","status":"completed","entries":[keep]}}})
    kwargs = dict(expected_model="m",expected_previous_version="old",reason="exclude unidentified refresh inputs")
    monkeypatch.setattr(outside_refresh_hull,"unidentified_refresh_inputs",lambda *a,**k:{})
    with pytest.raises(ValueError,match="evidence_missing"):
        control.rebuild_mlip_lineage(required_record_ids=["old"],**kwargs)
    assert read_json(path,{}) == state
    monkeypatch.setattr(outside_refresh_hull,"unidentified_refresh_inputs",lambda *a,**k:{("s","sha"):2})
    with pytest.raises(ValueError,match="scope_changed"):
        control.rebuild_mlip_lineage(required_record_ids=["keep"],**kwargs)
    report=control.rebuild_mlip_lineage(required_record_ids=["old"],**kwargs)
    saved=read_json(path,{})
    assert report["status"] == "rebuilt"
    assert saved["lineage_rebuild_history"][-1]["previous_diagram"] == state["phase_diagrams"]["mlip"]

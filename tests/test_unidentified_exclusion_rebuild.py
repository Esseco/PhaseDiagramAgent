from threading import RLock
from types import SimpleNamespace
import pytest
from phase_agent.runtime.local_agent_control import LocalAgentControl
from phase_agent.tools.step_runner.file_protocol import read_json, write_json


def test_rebuild_excludes_only_requested_unknown_and_keeps_history(tmp_path, monkeypatch):
    from phase_agent.analysis.phase import update_phase_diagram as module
    path = tmp_path / "state.json"
    unknown = {"record_id": "bad", "structure_id": "s", "phase": "Unidentified", "phase_identification_status": "unknown"}
    known = {"record_id": "good", "structure_id": "g", "phase": "O3", "phase_identification_status": "identified"}
    state = {"active_model_version": "m", "phase_records": [], "model_refresh": {"new_model_version": "m", "status": "completed"},
             "phase_diagrams": {"mlip": {"version": "old", "entries": [unknown, known]}},
             "phase_diagrams_by_model": {"m": {"version": "old-independent"}}}
    write_json(path, state)
    def fake(records, **kwargs):
        if kwargs.get("model_versions_only"):
            return {"mlip_by_version": {"m": {"status": "completed", "model_version": "m", "version": "independent", "entries": [known]}}}
        return {"diagrams": {"mlip": {"status": "completed", "model_version": "m", "version": "new", "entries": [known]}}}
    monkeypatch.setattr(module, "update_phase_diagram", fake)
    control = LocalAgentControl(SimpleNamespace(state_path=path, lock=RLock()))
    args = {"expected_model": "m", "expected_previous_version": "old", "reason": "exclude unidentified phase frames"}
    with pytest.raises(ValueError, match="scope_changed"):
        control.rebuild_mlip_lineage(**args, required_record_ids=["good"])
    assert read_json(path, {}) == state
    response = control.rebuild_mlip_lineage(**args, required_record_ids=["bad"])
    assert response["status"] == "rebuilt"
    saved = read_json(path, {})
    assert saved["phase_diagrams"]["mlip"]["entries"] == [known]
    assert saved["lineage_rebuild_history"][-1]["previous_diagram"]["entries"] == [unknown, known]
    assert saved["phase_diagrams_by_model"]["m"]["version"] == "independent"
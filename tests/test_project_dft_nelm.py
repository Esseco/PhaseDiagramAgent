from copy import deepcopy
from threading import RLock
from types import SimpleNamespace
import pytest

from phase_agent.runtime.local_agent_control import LocalAgentControl
from phase_agent.science.dft.incar_policy import layered_oxide_incar


def test_nelm_draft_preserves_confirmed_snapshot():
    session = {"status": "confirmed", "draft_revision": 4, "dialogue": [],
               "config": {"dft": {"parameters": {}}},
               "confirmed_snapshot": {"config_version": "old", "config": {"dft": {"parameters": {}}}}}
    handler = SimpleNamespace(state_path="state.json", lock=RLock(), workflow_kwargs={"config_session": deepcopy(session)})
    control = LocalAgentControl(handler)
    saved = []
    control._save_session = lambda value: saved.append(value)
    out = control.set_project_dft_nelm(150)
    assert out["confirmed"] is False and saved[0]["status"] == "draft"
    assert saved[0]["confirmed_snapshot"] == session["confirmed_snapshot"]
    assert saved[0]["config"]["dft"]["parameters"]["incar_settings"]["static"]["NELM"] == 150
    assert layered_oxide_incar({"NELM": 150}, static=True)["NELM"] == 150
    with pytest.raises(ValueError):
        control.set_project_dft_nelm(True)



def test_nelm_only_migration_preserves_old_tasks():
    from phase_agent.runtime.nelm_config_migration import migrate_state

    old = {"dft": {"parameters": {}}, "system": {"species": ["Ni"]}}
    state = {"confirmed_config_version": "old", "confirmed_config": old, "tasks": [{"status": "completed", "parameters": {"NELM": 120}}]}
    new = deepcopy(old)
    new["dft"]["parameters"]["incar_settings"] = {"static": {"NELM": 150}}
    snap = {"config_version": "new", "config": new}
    result = migrate_state(state, snap, "old", "new", 150)
    assert result["confirmed_config"] == new and result["tasks"] == state["tasks"]
    assert state["confirmed_config"] == old
    other = deepcopy(snap)
    other["config"]["system"]["species"] = ["Co"]
    with pytest.raises(ValueError, match="other_changes"):
        migrate_state(state, other, "old", "new", 150)
    with pytest.raises(ValueError, match="version_mismatch"):
        migrate_state(state, snap, "wrong", "new", 150)
    state["pending_execution_policies"] = {"x": {}}
    with pytest.raises(ValueError, match="active_work"):
        migrate_state(state, snap, "old", "new", 150)

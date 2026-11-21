from threading import RLock
from types import SimpleNamespace

import pytest

from phase_agent.runtime.local_agent_control import LocalAgentControl
from phase_agent.tools.step_runner.file_protocol import read_json, write_json


def test_retired_unidentified_inclusion_rebuild_is_rejected(tmp_path):
    path = tmp_path / "state.json"
    state = {"active_model_version": "m", "phase_diagrams": {"mlip": {"version": "old"}}}
    write_json(path, state)
    control = LocalAgentControl(SimpleNamespace(state_path=path, lock=RLock()))
    with pytest.raises(ValueError, match="lineage_rebuild_reason_invalid"):
        control.rebuild_mlip_lineage(expected_model="m", expected_previous_version="old", required_record_ids=["r"], reason="include verified unidentified MLIP frames")
    assert read_json(path, {}) == state

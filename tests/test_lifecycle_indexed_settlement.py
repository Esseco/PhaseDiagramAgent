"""Lifecycle reads only authoritative fields before input settlement when safe."""

import json

from phase_agent.graphs.project import runner
from phase_agent.tools.step_runner.file_protocol import read_json, write_json
from tests.test_training_return_contract import delivered


def test_no_delivery_avoids_full_startup_read(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    write_json(path, {"action_records": [], "history": ["x" * 1000] * 30})
    monkeypatch.setattr(
        runner,
        "_read_business_state",
        lambda _: (_ for _ in ()).throw(AssertionError("unnecessary full read")),
    )
    assert runner._settle_completed_input_delivery(path) is None


def test_legacy_state_without_index_uses_original_read(tmp_path):
    path = tmp_path / "state.json"
    state = {"action_records": [], "legacy": True}
    path.write_text(json.dumps(state), encoding="utf-8")
    assert runner._settle_completed_input_delivery(path) == state


def test_existing_training_delivery_still_settles_once(tmp_path):
    path = tmp_path / "state.json"
    state, action, execution = delivered()
    state["action_records"] = [
        {"record_id": "r", "final_action": action, "execution_result": execution}
    ]
    write_json(path, state)
    settled = runner._settle_completed_input_delivery(path)
    assert settled["budget_reservations"]["prepare"]["status"] == "settled"
    assert settled["effective_decisions"]["prepare"]["status"] == "completed"
    assert read_json(path) == settled
    assert runner._settle_completed_input_delivery(path) is None


def test_in_turn_unpersisted_state_uses_original_read(tmp_path, monkeypatch):
    from phase_agent.runtime import turn_process

    path = tmp_path / "state.json"
    write_json(path, {"action_records": []})
    expected = {"action_records": [], "unpersisted": True}
    monkeypatch.setattr(turn_process, "has_reusable_initial_state", lambda _: True)
    monkeypatch.setattr(runner, "_read_business_state", lambda _: expected)
    assert runner._settle_completed_input_delivery(path) is expected


def test_index_replaced_during_check_falls_back(tmp_path, monkeypatch):
    from phase_agent.persistence.state import field_index

    path = tmp_path / "state.json"
    write_json(path, {"action_records": []})
    original = field_index.IndexedState.verify
    count = 0

    def changed(self, *args, **kwargs):
        nonlocal count
        count += 1
        if count == 3:
            write_json(path, {"action_records": [], "updated": True})
        return original(self, *args, **kwargs)

    monkeypatch.setattr(field_index.IndexedState, "verify", changed)
    assert runner._settle_completed_input_delivery(path)["updated"] is True


def test_concurrent_process_cannot_overwrite_training_settlement(tmp_path, monkeypatch):
    import subprocess
    import sys

    import pytest
    from phase_agent.tools.workflows import training_input_settlement

    path = tmp_path / "state.json"
    state, action, execution = delivered()
    state["action_records"] = [
        {"record_id": "r", "final_action": action, "execution_result": execution}
    ]
    write_json(path, state)
    original = training_input_settlement.settle_saved_training_inputs

    def competing_write(current):
        result = original(current)
        script = (
            "import sys; from phase_agent.tools.step_runner.file_protocol import read_json, write_json; "
            "p=sys.argv[1]; s=read_json(p); s['other_process_update']=True; write_json(p,s)"
        )
        subprocess.run([sys.executable, "-c", script, str(path)], check=True)
        return result

    monkeypatch.setattr(training_input_settlement, "settle_saved_training_inputs", competing_write)
    with pytest.raises(ValueError, match="stale_state_file_version"):
        runner._settle_completed_input_delivery(path)
    saved = read_json(path)
    assert saved["other_process_update"] is True
    assert saved["budget_reservations"]["prepare"]["status"] == "reserved"
    assert saved["effective_decisions"]["prepare"]["status"] == "reserved"


def test_index_falls_back_after_same_metadata_field_rename(tmp_path):
    import subprocess
    import sys

    path = tmp_path / "state.json"
    write_json(path, {"action_records": [], "audit": "unchanged"})
    before = path.stat()
    script = (
        "import os,sys; from pathlib import Path; p=Path(sys.argv[1]); st=p.stat(); "
        "b=p.read_bytes(); old=b'\\\"action_records\\\"'; new=b'\\\"ignored_record\\\"'; "
        "assert old in b and len(old)==len(new); p.write_bytes(b.replace(old,new,1)); "
        "os.utime(p,ns=(st.st_atime_ns,st.st_mtime_ns))"
    )
    subprocess.run([sys.executable, "-c", script, str(path)], check=True)
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns, before.st_ino) == (
        after.st_size, after.st_mtime_ns, after.st_ino
    )
    recovered = runner._settle_completed_input_delivery(path)
    assert recovered["ignored_record"] == [] and "action_records" not in recovered


def test_index_falls_back_after_same_metadata_unread_field_change(tmp_path):
    from tests.test_strong_state_versions import _other_process_same_metadata

    path = tmp_path / "state.json"
    write_json(path, {"action_records": [], "audit": "A"})
    _other_process_same_metadata(path)
    recovered = runner._settle_completed_input_delivery(path)
    assert recovered["audit"] == "B"

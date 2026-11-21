from hashlib import sha256
import json
from pathlib import Path

import pytest

from phase_agent.tools.state.execution_receipts import (
    begin_execution,
    execution_events,
    record_business_saved,
    record_execution_return,
)
from phase_agent.tools.step_runner.file_protocol import write_json


IDENTITY = {
    "invocation_id": "a",
    "config_version": "c",
    "action_hash": "h",
    "tool": "generate_branches",
}


def business_state():
    return {
        "rows": [1, 2],
        "label": "中文",
        "invocations": {"a": {"status": "completed", "execution": {"status": "completed"}}},
    }


@pytest.mark.parametrize("legacy", [False, True])
def test_canonical_and_legacy_state_checks(tmp_path, legacy):
    path = tmp_path / "state.json"
    begin_execution(path, IDENTITY)
    record_execution_return(path, IDENTITY, "completed")
    state = business_state()
    if legacy:
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    else:
        write_json(path, state)
    with pytest.raises(ValueError, match="business state changed"):
        record_business_saved(path, {**state, "rows": [1, 3]})
    assert execution_events(path, "a")[-1]["event"] == "tool_returned"
    record_business_saved(path, state)
    saved = execution_events(path, "a")[-1]
    assert saved["event"] == "business_saved"
    assert saved["payload"]["state_file_sha256"] == sha256(path.read_bytes()).hexdigest()


def test_canonical_receipt_does_not_read_or_decode_full_state(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    begin_execution(path, IDENTITY)
    record_execution_return(path, IDENTITY, "completed")
    state = {**business_state(), "rows": ["x" * 10000] * 100}
    write_json(path, state)
    monkeypatch.setattr(
        Path, "read_bytes", lambda *_: (_ for _ in ()).throw(AssertionError("whole-file read"))
    )
    monkeypatch.setattr(
        json, "load", lambda *_: (_ for _ in ()).throw(AssertionError("unexpected decode"))
    )
    record_business_saved(path, state)
    assert execution_events(path, "a")[-1]["event"] == "business_saved"


@pytest.mark.parametrize("acknowledged", [False, True])
def test_no_new_acknowledgement_does_not_open_or_encode_state(tmp_path, monkeypatch, acknowledged):
    path = tmp_path / "state.json"
    begin_execution(path, IDENTITY)
    state = business_state()
    if acknowledged:
        record_execution_return(path, IDENTITY, "completed")
        write_json(path, state)
        record_business_saved(path, state)
    original = execution_events(path, "a")

    def forbidden(*args, **kwargs):
        raise AssertionError("no receipt is ready for acknowledgement")

    with monkeypatch.context() as guard:
        guard.setattr(Path, "open", forbidden)
        guard.setattr(json.JSONEncoder, "iterencode", forbidden)
        record_business_saved(path, state)
    assert execution_events(path, "a") == original
    assert not begin_execution(path, IDENTITY)["allowed"]

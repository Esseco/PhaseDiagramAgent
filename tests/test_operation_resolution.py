import hashlib
import json
from types import SimpleNamespace
import threading

import pytest
from phase_agent.runtime.local_agent_control import LocalAgentControl
from phase_agent.runtime.operation_resolution import resolve_operation
from phase_agent.tools.step_runner.build_status_summary import build_status_summary
from phase_agent.tools.step_runner.file_protocol import write_json


def setup(tmp_path):
    state = {"tasks": [], "confirmed_config_version": "new-config"}
    path = tmp_path / "state.json"
    write_json(path, state)
    control = LocalAgentControl(SimpleNamespace(state_path=path, lock=threading.RLock()))
    control.operations.directory.mkdir()
    record = control.operations.directory / ("a" * 32 + ".json")
    record.write_text("{broken")
    body = {
        "operation_id": "a" * 32,
        "record_sha256": hashlib.sha256(record.read_bytes()).hexdigest(),
        "state_version": build_status_summary(state, config_version="new-config")["summary_id"],
        "evidence_note": "read original receipts and outputs",
        "external_jobs": "none_found",
        "output_inventory": "checked_no_outputs",
    }
    return control, body, record


def test_resolution_retains_corrupt_original_and_never_replays_callback(tmp_path):
    control, body, record = setup(tmp_path)
    before = record.read_bytes()
    state = control.state_path.read_bytes()
    answer = control.resolve_operation(body)
    assert answer["status"] == "resolved" and not answer["replayed"] and not answer["submitted"]
    assert record.read_bytes() == before and control.state_path.read_bytes() == state
    assert control.operations.get(body["operation_id"])["status"] == "resolved"
    assert control.resolve_operation(body)["idempotent_replay"]
    calls = []
    from tests.test_ui_operations import wait

    first = control.operations.submit(
        lambda: calls.append(1) or {},
        request_id="new-explicit-interaction",
        request={"route": "/phase/propose"},
    )
    assert wait(control.operations, first["operation_id"])["status"] == "completed" and calls == [1]


@pytest.mark.parametrize(
    "fault",
    [
        "state_changed",
        "record_changed",
        "unsettled_receipt",
        "active_operation",
        "missing_attestation",
    ],
)
def test_resolution_fails_closed_on_stale_or_unverified_evidence(tmp_path, monkeypatch, fault):
    control, body, record = setup(tmp_path)
    if fault == "state_changed":
        body["state_version"] = "old"
    elif fault == "record_changed":
        record.write_text("{}")
    elif fault == "unsettled_receipt":
        monkeypatch.setattr(
            "phase_agent.tools.state.execution_receipts.recovery_report",
            lambda *a, **k: {"unsettled": True},
        )
    elif fault == "missing_attestation":
        body["evidence_note"] = ""
    else:
        # Inactive status must be established even with a correct file digest.
        monkeypatch.setattr(control.operations, "get", lambda identity: {"status": "running"})
    before = record.read_bytes()
    with pytest.raises(ValueError):
        control.resolve_operation(body)
    assert (
        record.read_bytes() == before
        and not (control.operations.directory / "resolutions").exists()
    )


def test_corrupt_resolution_isolated_and_preserved_when_reacknowledged(tmp_path):
    control, body, record = setup(tmp_path)
    control.resolve_operation(body)
    sidecar = control.operations.directory / "resolutions" / record.name
    sidecar.write_text("{damaged resolution")
    assert control.operations.get(body["operation_id"])["status"] == "unavailable"
    assert control.resolve_operation(body)["status"] == "resolved"
    backups = list((control.operations.directory / "resolution_backups").glob("*.json"))
    assert len(backups) == 1 and backups[0].read_text() == "{damaged resolution"


def test_http_resolution_works_even_when_async_submission_is_blocked(tmp_path):
    from http.client import HTTPConnection
    from phase_agent.runtime.agent_api import create_server

    control, body, record = setup(tmp_path)
    server = create_server(
        lambda *a, **k: None,
        api_key="a" * 24,
        control_api_key="c" * 24,
        local_control=control,
        port=0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = HTTPConnection(*server.server_address, timeout=3)
    try:
        connection.request(
            "POST",
            "/phase/operations/resolve",
            json.dumps(body),
            {
                "Authorization": "Bearer " + "c" * 24,
                "Content-Type": "application/json",
                "Prefer": "respond-async",
                "Idempotency-Key": "not-a-replay",
            },
        )
        response = connection.getresponse()
        answer = json.loads(response.read())
        assert response.status == 200 and answer["status"] == "resolved" and not answer["replayed"]
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(3)

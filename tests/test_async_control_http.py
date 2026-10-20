from http.client import HTTPConnection
import json
import threading
from types import SimpleNamespace

from phase_agent.runtime.agent_api import create_server
from phase_agent.runtime.ui_operations import UIOperations
from tests.test_ui_operations import wait


def test_async_post_remains_exact_operation_on_reused_poll_connection(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    calls = []
    store = UIOperations(tmp_path / "state.json", threading.RLock())
    run = store._run

    def delayed(identity, callback):
        entered.set()
        assert release.wait(3)
        run(identity, callback)

    monkeypatch.setattr(store, "_run", delayed)
    control = SimpleNamespace(
        operations=store,
        recovery_transaction=lambda: {"status": "none"},
        propose=lambda instruction, **kw: calls.append(instruction) or {"reply": "proposal"},
        pause=lambda *args, **kw: (_ for _ in ()).throw(AssertionError("route changed")),
    )
    server = create_server(
        lambda *a, **kw: None,
        api_key="a" * 24,
        control_api_key="c" * 24,
        local_control=control,
        port=0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = HTTPConnection(*server.server_address, timeout=3)
    headers = {
        "Authorization": "Bearer " + "c" * 24,
        "Content-Type": "application/json",
        "Prefer": "respond-async",
        "Idempotency-Key": "unique",
    }
    try:
        connection.request("POST", "/phase/propose", json.dumps({"instruction": "review"}), headers)
        response = connection.getresponse()
        assert response.status == 202
        operation = json.loads(response.read())
        assert entered.wait(1)
        connection.request(
            "GET", operation["status_url"], headers={"Authorization": "Bearer " + "c" * 24}
        )
        response = connection.getresponse()
        assert response.status == 200
        response.read()
        release.set()
        done = wait(store, operation["operation_id"])
        assert (
            done["status"] == "completed"
            and done["result"] == {"reply": "proposal"}
            and calls == ["review"]
        )
        connection.request("POST", "/phase/propose", json.dumps({"instruction": "review"}), headers)
        response = connection.getresponse()
        assert response.status == 202
        assert json.loads(response.read())["idempotent_replay"]
        assert calls == ["review"]
        connection.request(
            "GET", "/phase/recovery/transaction", headers={"Authorization": "Bearer " + "c" * 24}
        )
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read()) == {"status": "none"}
        control.recovery_transaction = lambda: {"status": "interrupted", "revision": "saved"}
        connection.request(
            "GET", "/phase/recovery/transaction", headers={"Authorization": "Bearer " + "c" * 24}
        )
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read())["revision"] == "saved"
        connection.request("GET", "/phase/recovery/transaction")
        response = connection.getresponse()
        assert response.status == 401
        response.read()
        connection.request("GET", operation["status_url"])
        response = connection.getresponse()
        assert response.status == 401
        response.read()
    finally:
        release.set()
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_transaction_resume_is_available_when_uncertain_ui_request_blocks_queue():
    calls = []

    def forbidden(*a, **kw):
        raise AssertionError("transaction resume must not queue behind interrupted operation")

    control = SimpleNamespace(
        operations=SimpleNamespace(submit=forbidden),
        resume_recovery_transaction=lambda revision: (
            calls.append(revision) or {"status": "completed", "submitted": False}
        ),
    )
    server = create_server(
        lambda *a, **kw: None,
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
            "/phase/recovery/transaction/resume",
            json.dumps({"revision": "original-approved-revision"}),
            {
                "Authorization": "Bearer " + "c" * 24,
                "Content-Type": "application/json",
                "Prefer": "respond-async",
            },
        )
        response = connection.getresponse()
        answer = json.loads(response.read())
        assert (
            response.status == 200
            and answer["status"] == "completed"
            and calls == ["original-approved-revision"]
        )
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(3)

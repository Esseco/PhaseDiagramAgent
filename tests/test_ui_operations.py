import json
from pathlib import Path
import threading
import time

import pytest

from phase_agent.runtime.ui_operations import UIOperations


def wait(store, identity):
    for n in range(100):
        result = store.get(identity)
        if result["status"] not in {"pending", "running"}:
            return result
        time.sleep(0.01)
    raise AssertionError("operation did not finish")


def test_background_request_returns_before_work_and_duplicate_never_replays(tmp_path):
    store = UIOperations(tmp_path / "state.json", threading.RLock())
    release = threading.Event()
    started = threading.Event()
    calls = []

    def callback():
        calls.append(1)
        started.set()
        release.wait(2)
        return {"status": "awaiting_approval", "state": {"huge": "not a UI result"}}

    first = store.submit(
        callback, request_id="one", request={"route": "/phase/propose", "body": {}}
    )
    assert first["status"] == "pending" and started.wait(1)
    duplicate = store.submit(
        callback, request_id="one", request={"route": "/phase/propose", "body": {}}
    )
    assert duplicate["idempotent_replay"]
    other = store.submit(
        callback, request_id="two", request={"route": "/phase/propose", "body": {}}
    )
    assert other["busy"] and other["operation_id"] == first["operation_id"]
    release.set()
    done = wait(store, first["operation_id"])
    assert (
        done["status"] == "completed"
        and done["result"] == {"status": "awaiting_approval"}
        and len(calls) == 1
    )
    assert store.submit(
        callback, request_id="one", request={"route": "/phase/propose", "body": {}}
    )["idempotent_replay"]


def test_busy_studio_blocks_mutation_without_calling_it(tmp_path):
    lock = threading.RLock()
    store = UIOperations(tmp_path / "state.json", lock)
    calls = []
    with lock:
        operation = store.submit(
            lambda: calls.append(1), request_id="busy", request={"route": "/phase/decision"}
        )
        done = wait(store, operation["operation_id"])
    assert done["status"] == "blocked" and calls == []


def test_changed_request_cannot_reuse_idempotency_key(tmp_path):
    store = UIOperations(tmp_path / "state.json", threading.RLock())
    first = store.submit(
        lambda: {"status": "done"},
        request_id="id",
        request={"route": "/phase/propose", "body": {"instruction": "A"}},
    )
    wait(store, first["operation_id"])
    with pytest.raises(ValueError, match="different_request"):
        store.submit(
            lambda: None,
            request_id="id",
            request={"route": "/phase/propose", "body": {"instruction": "B"}},
        )


def test_operation_queries_are_read_only_and_reject_path_escape(tmp_path):
    store = UIOperations(tmp_path / "state.json", threading.RLock())
    first = store.submit(
        lambda: {"status": "done"}, request_id="id", request={"route": "/phase/propose"}
    )
    wait(store, first["operation_id"])
    path = store._path(first["operation_id"])
    before = path.read_bytes()
    assert store.get(first["operation_id"])["status"] == "completed"
    assert path.read_bytes() == before
    with pytest.raises(ValueError):
        store.get("../state")


def test_dead_owner_is_reported_interrupted_without_replaying_callback(tmp_path):
    store = UIOperations(tmp_path / "state.json", threading.RLock())
    first = store.submit(
        lambda: {"status": "done"}, request_id="id", request={"route": "/phase/propose"}
    )
    wait(store, first["operation_id"])
    path = store._path(first["operation_id"])
    record = json.loads(path.read_text())
    record.update(status="running", owner_pid=2147483647, owner_started=0.0)
    path.write_text(json.dumps(record))
    before = path.read_bytes()
    assert store.get(first["operation_id"])["status"] == "interrupted"
    assert path.read_bytes() == before
    calls = []
    replay = store.submit(
        lambda: calls.append(1), request_id="id", request={"route": "/phase/propose"}
    )
    assert replay["status"] == "interrupted" and calls == []

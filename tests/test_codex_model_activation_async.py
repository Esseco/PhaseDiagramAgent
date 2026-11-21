import threading
import time
from types import SimpleNamespace

import pytest

from phase_agent.runtime.codex.mcp import CodexMcpServer
from phase_agent.runtime.ui_operations import UIOperations


def test_activation_is_tracked_and_not_replayed(tmp_path):
    started, release = threading.Event(), threading.Event()
    calls = []
    operations = UIOperations(tmp_path / "state.json", threading.RLock())

    class Fake:
        control = SimpleNamespace(operations=operations)

        def decide(self, decision, **kwargs):
            calls.append(kwargs["plan_id"])
            started.set()
            assert release.wait(10)
            return {"status": "activated"}

    server = CodexMcpServer(Fake())
    args = {
        "decision": "confirm_sensitive",
        "plan_id": "review:model_activation:abc",
        "state_version": "v1",
        "proposal_hash": "sha256:abc",
        "comment": "approved",
    }
    try:
        result = server._call_tool("phase_decide", args)
        assert result["status"] == "pending"
        assert started.wait(3)
        duplicate = server._call_tool("phase_decide", args)
        assert duplicate["operation_id"] == result["operation_id"]
        assert duplicate["idempotent_replay"] is True
        with pytest.raises(ValueError, match="idempotency_key_reused"):
            server._call_tool("phase_decide", {**args, "comment": "changed"})
        assert len(calls) == 1
    finally:
        release.set()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        status = server._call_tool(
            "phase_operation_status", {"operation_id": result["operation_id"]}
        )
        if status["status"] not in {"running", "pending"}:
            break
        time.sleep(0.02)
    assert status["status"] == "completed"
    assert status["result"]["status"] == "activated"
    assert len(calls) == 1


def test_failed_activation_is_recorded_without_replay(tmp_path, monkeypatch):
    operations = UIOperations(tmp_path / "state.json", threading.RLock())
    calls = []
    monkeypatch.setattr(
        "phase_agent.runtime.boundary_diagnostics.record_failure",
        lambda *args, **kwargs: {"message": "failed", "execution_effect": "unknown"},
    )

    class Fake:
        control = SimpleNamespace(operations=operations)

        def decide(self, decision, **kwargs):
            calls.append(kwargs["plan_id"])
            raise RuntimeError("simulated interruption")

    server = CodexMcpServer(Fake())
    args = {
        "decision": "confirm_sensitive",
        "plan_id": "review:model_activation:failure",
        "state_version": "v1",
        "proposal_hash": "sha256:failure",
    }
    result = server._call_tool("phase_decide", args)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        status = server._call_tool(
            "phase_operation_status", {"operation_id": result["operation_id"]}
        )
        if status["status"] not in {"running", "pending"}:
            break
        time.sleep(0.02)
    assert status["status"] == "blocked"
    assert server._call_tool("phase_decide", args)["idempotent_replay"] is True
    assert len(calls) == 1


def test_refresh_approval_also_uses_background(tmp_path):
    release = threading.Event()
    operations = UIOperations(tmp_path / "state.json", threading.RLock())

    class Fake:
        control = SimpleNamespace(operations=operations)

        def decide(self, decision, **kwargs):
            assert release.wait(10)
            return {"status": "prepared"}

    server = CodexMcpServer(Fake())
    args = {
        "decision": "approve",
        "plan_id": "webui-refresh",
        "state_version": "v1",
        "proposal_hash": "sha256:refresh",
    }
    try:
        result = server._call_tool("phase_decide", args)
        assert result["status"] == "pending"
        assert server._call_tool("phase_decide", args)["idempotent_replay"] is True
    finally:
        release.set()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        status = server._call_tool(
            "phase_operation_status", {"operation_id": result["operation_id"]}
        )
        if status["status"] not in {"running", "pending"}:
            break
        time.sleep(0.02)
    assert status["status"] == "completed"
    assert status["result"]["status"] == "prepared"

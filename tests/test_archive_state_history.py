"""Tests for phase_agent.persistence.state.archive_history."""

from __future__ import annotations

import hashlib
import json

from phase_agent.persistence.state.archive_history import archive_state_history, main
from phase_agent.tools.step_runner.file_protocol import read_json, write_json


def _sample_state() -> dict:
    return {
        "epoch": 2,
        "tasks": [{"task_id": "t1", "status": "completed"}],
        "pending_execution_policies": {
            "pol-1": {"record_id": "pol-1", "decision_id": "dec-1", "status": "pending"}
        },
        "execution_reconciliations": {"rec-1": {"action_id": "act-1"}},
        "event_history": [{"id": f"ev-{i}", "status": "completed"} for i in range(10)],
        "decisions": [
            {"decision_id": "dec-0", "status": "completed"},
            {"decision_id": "dec-1", "status": "completed"},  # referenced by pending policy
            {"decision_id": "dec-2", "status": "pending"},  # active status
            {"decision_id": "dec-3", "status": "completed"},
            {"decision_id": "dec-4", "status": "completed"},
        ],
        "invocations": [{"invocation_id": f"inv-{i}", "status": "completed"} for i in range(6)],
        "action_records": [
            {"action_id": "act-0", "status": "completed"},
            {"action_id": "act-1", "status": "completed"},  # referenced by reconciliation
            {"action_id": "act-2", "status": "completed"},
            {"action_id": "act-3", "status": "completed"},
        ],
        "cost_history": [{"id": f"cost-{i}", "amount": 1.0} for i in range(5)],
    }


def test_archive_state_history_preserves_protected_and_verifies_sha256(tmp_path):
    state_path = tmp_path / "workflow_state" / "state.json"
    original = _sample_state()
    write_json(state_path, original)

    result = archive_state_history(
        state_path,
        keep_recent_events=2,
        keep_recent_decisions=1,
        keep_recent_invocations=2,
        keep_recent_actions=1,
        keep_recent_costs=2,
    )

    assert result["status"] == "archived"
    archive_path = tmp_path / "workflow_state" / "history"
    assert result["archive_path"].startswith(str(archive_path))
    raw_archive = open(result["archive_path"], "rb").read()
    assert hashlib.sha256(raw_archive).hexdigest() == result["archive_sha256"]

    archived_payload = json.loads(raw_archive.decode("utf-8"))
    slimmed = read_json(state_path)

    # Protected records (dec-1 via pending policy, dec-2 via pending status, act-1 via reconciliation) stay in state.json.
    kept_decision_ids = [row["decision_id"] for row in slimmed["decisions"]]
    assert kept_decision_ids == ["dec-1", "dec-2", "dec-4"]
    archived_decision_ids = [row["decision_id"] for row in archived_payload["records"]["decisions"]]
    assert archived_decision_ids == ["dec-0", "dec-3"]

    kept_action_ids = [row["action_id"] for row in slimmed["action_records"]]
    assert kept_action_ids == ["act-1", "act-3"]
    archived_action_ids = [
        row["action_id"] for row in archived_payload["records"]["action_records"]
    ]
    assert archived_action_ids == ["act-0", "act-2"]

    assert [row["id"] for row in slimmed["event_history"]] == ["ev-8", "ev-9"]
    assert len(archived_payload["records"]["event_history"]) == 8
    assert [row["invocation_id"] for row in slimmed["invocations"]] == ["inv-4", "inv-5"]
    assert len(archived_payload["records"]["invocations"]) == 4
    assert [row["id"] for row in slimmed["cost_history"]] == ["cost-3", "cost-4"]
    assert len(archived_payload["records"]["cost_history"]) == 3

    assert len(slimmed["history_archives"]) == 1
    assert slimmed["history_archives"][0]["sha256"] == result["archive_sha256"]
    assert slimmed["tasks"] == original["tasks"]


def test_archive_state_history_dry_run_and_noop(tmp_path):
    state_path = tmp_path / "workflow_state" / "state.json"
    original = _sample_state()
    write_json(state_path, original)
    before_bytes = state_path.read_bytes()

    dry = archive_state_history(state_path, keep_recent_events=2, dry_run=True)
    assert dry["status"] == "dry_run"
    assert dry["archive_path"] is None
    assert dry["archived_counts"]["event_history"] == 8
    assert state_path.read_bytes() == before_bytes

    noop = archive_state_history(
        state_path,
        keep_recent_events=100,
        keep_recent_decisions=100,
        keep_recent_invocations=100,
        keep_recent_actions=100,
        keep_recent_costs=100,
    )
    assert noop["status"] == "noop"
    assert noop["archive_path"] is None


def test_archive_state_history_cli(tmp_path, capsys):
    state_path = tmp_path / "workflow_state" / "state.json"
    write_json(state_path, _sample_state())
    rc = main(["--state", str(state_path), "--keep-events", "3", "--dry-run"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "dry_run"
    assert out["archived_counts"]["event_history"] == 7

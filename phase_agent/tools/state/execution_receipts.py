"""Persistent side-effect boundary; never interpret a receipt as a result."""

import json
import os
from pathlib import Path
import sqlite3
from phase_agent.graphs.contracts import ExecutionIdentity


def _identity(state_path, identity):
    validated = ExecutionIdentity.model_validate(identity).model_dump()
    owner = os.path.normcase(str(Path(state_path).resolve()))
    key = json.dumps([owner, validated["invocation_id"]], ensure_ascii=False)
    return key, json.dumps(validated, sort_keys=True, ensure_ascii=False)


def _connect(state_path):
    path = Path(state_path).resolve().parent / "execution_receipts.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS receipts "
        "(action_id TEXT PRIMARY KEY, identity TEXT NOT NULL, "
        "phase TEXT NOT NULL, outcome TEXT)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS execution_events "
        "(action_id TEXT NOT NULL, event TEXT NOT NULL, payload TEXT NOT NULL, "
        "created_at TEXT NOT NULL, PRIMARY KEY(action_id, event))"
    )
    connection.commit()
    return connection


def begin_execution(state_path, identity, *, action=None):
    """Atomically claim once. Any existing identity requires business reconciliation."""
    key, serialized = _identity(state_path, identity)
    connection = _connect(state_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        previous = connection.execute(
            "SELECT identity, phase FROM receipts WHERE action_id=?", (key,)
        ).fetchone()
        if previous:
            connection.rollback()
            return {
                "allowed": False,
                "reason": "identity_changed"
                if previous[0] != serialized
                else "execution_unsettled",
                "phase": previous[1],
            }
        connection.execute("INSERT INTO receipts VALUES (?, ?, 'started', NULL)", (key, serialized))
        _event(connection, key, "claimed", {"identity": identity, "action": action or {}})
        connection.commit()
        return {"allowed": True, "phase": "started"}
    finally:
        connection.close()


def record_execution_return(state_path, identity, outcome, *, evidence=None):
    """The tool returned; this is not proof of business ledger settlement."""
    key, serialized = _identity(state_path, identity)
    if not isinstance(outcome, str) or not outcome:
        raise ValueError("execution outcome must be a nonempty status")
    connection = _connect(state_path)
    try:
        cursor = connection.execute(
            "UPDATE receipts SET phase='returned', outcome=? "
            "WHERE action_id=? AND identity=? AND phase='started'",
            (outcome, key, serialized),
        )
        if cursor.rowcount != 1:
            connection.rollback()
            raise ValueError("execution receipt missing or identity changed")
        _event(
            connection,
            key,
            "tool_returned",
            {"identity": identity, "outcome": outcome, "evidence": evidence or {}},
        )
        connection.commit()
    finally:
        connection.close()


def recovery_report(state_path, state):
    """Read-only startup reconciliation; never infer completion from tool return."""
    from phase_agent.tools.state.execution_consistency import execution_state_issues

    unsettled = execution_state_issues(state)
    path = Path(state_path).resolve().parent / "execution_receipts.sqlite"
    if not path.exists():
        return {
            "status": "reconciliation_required" if unsettled else "clear",
            "unsettled": unsettled,
        }
    owner = os.path.normcase(str(Path(state_path).resolve()))
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT action_id, identity, phase, outcome FROM receipts"
        ).fetchall()
    finally:
        connection.close()
    for key, identity, phase, outcome in rows:
        receipt_owner, invocation = json.loads(key)
        if receipt_owner != owner:
            continue
        audited = (state.get("execution_reconciliations") or {}).get(invocation) or {}
        if (
            audited.get("resolution") in {"verified_no_effect", "verified_registered_effects"}
            and audited.get("identity") == json.loads(identity)
            and audited.get("evidence")
            and audited.get("reviewed_at")
        ):
            # The original receipt stays claimed: never replay that interrupted action.
            continue
        business = (state.get("invocations") or {}).get(invocation) or {}
        # A reconciliation response is not evidence of completion.
        execution = business.get("execution") or business.get("execution_result") or {}
        if (
            phase == "returned"
            and execution.get("status") == outcome
            and outcome in {"completed", "failed"}
            and business.get("status") not in {None, "execution_reconciliation_required"}
        ):
            continue
        unsettled.append(
            {
                "identity": json.loads(identity),
                "phase": phase,
                "tool_outcome": outcome,
                "required_action": "reconcile_business_artifacts",
            }
        )
    return {"status": "reconciliation_required" if unsettled else "clear", "unsettled": unsettled}


def _event(connection, key, event, payload):
    from datetime import datetime, timezone

    connection.execute(
        "INSERT OR IGNORE INTO execution_events VALUES (?, ?, ?, ?)",
        (
            key,
            event,
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            datetime.now(timezone.utc).isoformat(),
        ),
    )


def execution_events(state_path, invocation_id):
    """Read old/new journals without creating or upgrading databases."""
    path = Path(state_path).resolve().parent / "execution_receipts.sqlite"
    if not path.exists():
        return []
    key = json.dumps(
        [os.path.normcase(str(Path(state_path).resolve())), invocation_id], ensure_ascii=False
    )
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    try:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='execution_events'"
        ).fetchone()
        if not exists:
            return []
        return [
            {"event": event, "payload": json.loads(payload), "created_at": timestamp}
            for event, payload, timestamp in connection.execute(
                "SELECT event,payload,created_at FROM execution_events WHERE action_id=? ORDER BY rowid",
                (key,),
            )
        ]
    finally:
        connection.close()


def record_business_saved(state_path, state):
    """Acknowledge only outcomes already present in the durably saved JSON."""
    path = Path(state_path).resolve().parent / "execution_receipts.sqlite"
    if not path.exists():
        return
    import hashlib

    saved_bytes = Path(state_path).read_bytes()
    saved_state = json.loads(saved_bytes)
    if saved_state != json.loads(json.dumps(state, default=str)):
        raise ValueError("business state changed before settlement acknowledgement")
    digest = hashlib.sha256(saved_bytes).hexdigest()
    owner = os.path.normcase(str(Path(state_path).resolve()))
    connection = _connect(state_path)
    try:
        for key, identity, outcome in connection.execute(
            "SELECT action_id,identity,outcome FROM receipts WHERE phase='returned'"
        ).fetchall():
            saved_owner, invocation = json.loads(key)
            business = (saved_state.get("invocations") or {}).get(invocation) or {}
            execution = business.get("execution") or business.get("execution_result") or {}
            if (
                saved_owner == owner
                and outcome in {"completed", "failed"}
                and execution.get("status") == outcome
                and business.get("status") not in {None, "execution_reconciliation_required"}
            ):
                _event(
                    connection,
                    key,
                    "business_saved",
                    {
                        "identity": json.loads(identity),
                        "status": business["status"],
                        "state_file_sha256": digest,
                    },
                )
        connection.commit()
    finally:
        connection.close()

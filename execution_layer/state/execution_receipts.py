"""Persistent side-effect boundary; never interpret a receipt as a result."""
import json
import os
from pathlib import Path
import sqlite3
from orchestration.contracts import ExecutionIdentity


def _identity(state_path, identity):
    validated = ExecutionIdentity.model_validate(identity).model_dump()
    owner = os.path.normcase(str(Path(state_path).resolve()))
    key = json.dumps([owner, validated["invocation_id"]], ensure_ascii=False)
    return key, json.dumps(validated, sort_keys=True, ensure_ascii=False)


def _connect(state_path):
    path = Path(state_path).resolve().parent / "execution_receipts.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.execute("CREATE TABLE IF NOT EXISTS receipts "
                       "(action_id TEXT PRIMARY KEY, identity TEXT NOT NULL, "
                       "phase TEXT NOT NULL, outcome TEXT)")
    connection.commit()
    return connection


def begin_execution(state_path, identity):
    """Atomically claim once. Any existing identity requires business reconciliation."""
    key, serialized = _identity(state_path, identity)
    connection = _connect(state_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        previous = connection.execute(
            "SELECT identity, phase FROM receipts WHERE action_id=?", (key,)).fetchone()
        if previous:
            connection.rollback()
            return {"allowed": False, "reason": "identity_changed" if previous[0] != serialized
                    else "execution_unsettled", "phase": previous[1]}
        connection.execute("INSERT INTO receipts VALUES (?, ?, 'started', NULL)", (key, serialized))
        connection.commit()
        return {"allowed": True, "phase": "started"}
    finally:
        connection.close()


def record_execution_return(state_path, identity, outcome):
    """The tool returned; this is not proof of business ledger settlement."""
    key, serialized = _identity(state_path, identity)
    if not isinstance(outcome, str) or not outcome:
        raise ValueError("execution outcome must be a nonempty status")
    connection = _connect(state_path)
    try:
        cursor = connection.execute(
            "UPDATE receipts SET phase='returned', outcome=? "
            "WHERE action_id=? AND identity=? AND phase='started'", (outcome, key, serialized))
        if cursor.rowcount != 1:
            connection.rollback()
            raise ValueError("execution receipt missing or identity changed")
        connection.commit()
    finally:
        connection.close()


def recovery_report(state_path, state):
    """Read-only startup reconciliation; never infer completion from tool return."""
    path = Path(state_path).resolve().parent / "execution_receipts.sqlite"
    if not path.exists():
        return {"status": "clear", "unsettled": []}
    owner = os.path.normcase(str(Path(state_path).resolve()))
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    try:
        rows = connection.execute("SELECT action_id, identity, phase, outcome FROM receipts").fetchall()
    finally:
        connection.close()
    unsettled = []
    for key, identity, phase, outcome in rows:
        receipt_owner, invocation = json.loads(key)
        if receipt_owner != owner:
            continue
        business = (state.get("invocations") or {}).get(invocation) or {}
        # A reconciliation response is not evidence of completion.
        execution = business.get("execution") or business.get("execution_result") or {}
        if (phase == "returned" and execution.get("status") == outcome
                and outcome in {"completed", "failed"}
                and business.get("status") not in {None, "execution_reconciliation_required"}):
            continue
        unsettled.append({"identity": json.loads(identity), "phase": phase,
                          "tool_outcome": outcome, "required_action": "reconcile_business_artifacts"})
    return {"status": "reconciliation_required" if unsettled else "clear", "unsettled": unsettled}

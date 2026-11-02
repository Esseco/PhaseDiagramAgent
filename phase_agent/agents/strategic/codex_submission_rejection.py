"""Verified checkpoint proof for a proposal rejected before scientific execution."""

import hashlib
import json
from contextlib import closing
from pathlib import Path
import sqlite3

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from phase_agent.persistence.compact_checkpoints import CompressedSerde
from phase_agent.persistence.sqlite_paths import readonly_uri


def rejection_checkpoint(state_path, session, *, checkpoint_id, request_id, checkpoint_sha256):
    submission = session.get("submission") or {}
    reason = submission.get("reason") or ""
    marker = "llm_failed: ValueError: Invalid budget trade-off: "
    from phase_agent.agents.strategic.round_budget_review import round_budget_review_errors

    errors = round_budget_review_errors(
        session["draft"]["action"], {"round_budget_evidence": {"required": True}}
    )
    proven_reason = any(marker + error in reason for error in errors)
    if session["draft"]["action"].get("scientific_decision") is None:
        from phase_agent.agents.strategic.scientific_decisions import scientific_decision_errors

        schema_errors = scientific_decision_errors(session["draft"]["action"], {}, {})
        proven_reason = proven_reason or any(
            "llm_failed: ValueError: Invalid scientific decision: " + error in reason
            for error in schema_errors
        )
    # Only reconcile this exact branch-review rejection when the unknown
    # identifiers were actually cited by the frozen action. Checkpoint and
    # execution-receipt verification below remain mandatory.
    marker = "branch_review_invalid_after_one_correction: branch_pool_review.evidence_refs: unknown source "
    if marker in reason:
        import ast
        import re

        match = re.search(re.escape(marker) + r"(\[[^\]]*\])", reason)
        action = session["draft"]["action"]
        cited = (action.get("branch_pool_review") or {}).get("evidence_refs") or []
        try:
            unknown = ast.literal_eval(match.group(1)) if match else None
        except (ValueError, SyntaxError):
            unknown = None
        proven_reason = proven_reason or (
            action.get("tool") == "generate_branches"
            and isinstance(unknown, list)
            and bool(unknown)
            and all(
                isinstance(ref, str) and ref.startswith("branch_pool:") and ref in cited
                for ref in unknown
            )
        )
    if submission.get("status") != "not_configured" or not proven_reason:
        raise ValueError("submission_has_no_proven_proposal_validation_failure")
    path = Path(state_path).resolve()
    with closing(
        sqlite3.connect(readonly_uri(path.parent / "langgraph_lifecycle.sqlite"), uri=True)
    ) as conn:
        row = conn.execute(
            "SELECT thread_id,checkpoint_id,type,checkpoint FROM checkpoints "
            "WHERE checkpoint_ns='' ORDER BY checkpoint_id DESC LIMIT 1"
        ).fetchone()
        if (
            not row
            or row[1] != checkpoint_id
            or hashlib.sha256(row[3]).hexdigest() != checkpoint_sha256
        ):
            raise ValueError("submission_checkpoint_changed")
        payload = CompressedSerde(JsonPlusSerializer()).loads_typed(
            (row[2].removeprefix("phase-cas-v1:"), row[3])
        )
    values = payload.get("channel_values") or {}
    response = values.get("completed_response") or {}
    frame = values.get("business_frame") or {}
    if (
        values.get("request_id") != request_id
        or (session.get("dispatch_request_id") not in {None, request_id})
        or values.get("phase") != "finalized"
        or values.get("current_node") != "prepare_inputs_and_finalize"
        or values.get("node_status") != "completed"
        or Path(frame.get("runtime_state_path") or "").resolve() != path
        or response.get("status") != "not_configured"
        or response.get("submitted") is not False
        or response.get("reason") != reason
        or any(
            response.get(k)
            for k in ("record_id", "agent_proposal", "execution", "execution_result")
        )
    ):
        raise ValueError("submission_checkpoint_does_not_prove_zero_execution")
    events = response.get("events")
    if (
        not isinstance(events, list)
        or len(events) != 1
        or events[0].get("status") != "not_configured"
        or events[0].get("reason") != reason
    ):
        raise ValueError("submission_checkpoint_has_other_events")
    receipt_path = path.parent / "execution_receipts.sqlite"
    if receipt_path.exists():
        with closing(sqlite3.connect(readonly_uri(receipt_path), uri=True)) as conn:
            if any(
                json.loads(identity).get("invocation_id") == request_id
                for (identity,) in conn.execute("SELECT identity FROM receipts")
            ):
                raise ValueError("submission_has_execution_receipt")
    return {
        "checkpoint_id": checkpoint_id,
        "checkpoint_sha256": checkpoint_sha256,
        "request_id": request_id,
        "thread_id": row[0],
        "classification": "proposal_validation_rejected",
        "executed": False,
    }

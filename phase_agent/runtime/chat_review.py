"""Versioned approval routing; native persistence is handled by workflow policy."""

from phase_agent.tools.step_runner.file_protocol import read_json
from phase_agent.tools.step_runner.build_status_summary import build_status_summary


def review_pending(
    handler,
    plan_id,
    decision,
    *,
    expected_state_version,
    expected_proposal_hash,
    comment="",
    request_error=ValueError,
    is_sensitive=None,
):
    """Accept a decision only from the authenticated local approval surface."""
    from phase_agent.tools.policy.file_approval import proposal_hash

    with handler.lock:
        state = read_json(handler.state_path, {}) or {}
        completed = (state.get("invocations") or {}).get(plan_id)
        pending = (state.get("pending_execution_policies") or {}).get(plan_id)
        if pending is None and completed is not None:
            return {"status": "already_processed", "result": completed}
        if pending is None:
            raise request_error("plan is not pending")
        current_version = build_status_summary(
            state, config_version=state.get("confirmed_config_version")
        )["summary_id"]
        if expected_state_version != current_version:
            raise request_error("state_version_stale; refresh the approval page")
        proposal = pending.get("agent_proposal") or {}
        if expected_proposal_hash != proposal_hash(proposal):
            raise request_error("proposal_hash_mismatch; refresh the approval page")
        if decision not in {"approve", "reject", "confirm_sensitive"}:
            raise request_error("invalid approval-page decision")
        if is_sensitive(proposal) and decision == "approve":
            raise request_error("sensitive action requires confirm_sensitive")
        approved = decision in {"approve", "confirm_sensitive"}
        audit_comment = str(comment or "").strip()
        if approved:
            audit_comment = (audit_comment + "\napprove").strip()
        feedback = {"decision": "approve" if approved else "reject", "comment": audit_comment}
        result = handler._run(plan_id, feedback, "local approval page")
        return {"status": result.get("status"), "result": result}

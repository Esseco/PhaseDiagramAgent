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
            from phase_agent.runtime.review_requests import additional_reviews, review_additional

            rows = additional_reviews(state, handler.workflow_kwargs.get("config_session"))
            row = next((item for item in rows if item["plan_id"] == plan_id), None)
            if row is None:
                raise request_error("plan is not pending")
            version = build_status_summary(
                state, config_version=state.get("confirmed_config_version")
            )["summary_id"]
            if expected_state_version != version or expected_proposal_hash != row["proposal_hash"]:
                raise request_error("review_changed; refresh the approval page")
            if decision not in {"approve", "reject", "confirm_sensitive", "modify"}:
                raise request_error("invalid approval-page decision")
            return review_additional(handler, state, row, decision, comment)
        current_version = build_status_summary(
            state, config_version=state.get("confirmed_config_version")
        )["summary_id"]
        if expected_state_version != current_version:
            raise request_error("state_version_stale; refresh the approval page")
        proposal = pending.get("agent_proposal") or {}
        if expected_proposal_hash != proposal_hash(proposal):
            raise request_error("proposal_hash_mismatch; refresh the approval page")
        if decision not in {"approve", "reject", "confirm_sensitive", "modify"}:
            raise request_error("invalid approval-page decision")
        if is_sensitive(proposal) and decision == "approve":
            raise request_error("sensitive action requires confirm_sensitive")
        if decision == "modify":
            instruction = str(comment or "").strip()
            if not instruction:
                raise request_error("请填写修改要求；原方案未改变")
            result = handler._run(
                plan_id, {"decision": "comment", "comment": instruction}, instruction
            )
            return {"status": result.get("status"), "result": result}
        approved = decision in {"approve", "confirm_sensitive"}
        audit_comment = str(comment or "").strip()
        if approved:
            audit_comment = (audit_comment + "\napprove").strip()
        feedback = {"decision": "approve" if approved else "reject", "comment": audit_comment}
        result = handler._run(plan_id, feedback, "local approval page")
        return {"status": result.get("status"), "result": result}

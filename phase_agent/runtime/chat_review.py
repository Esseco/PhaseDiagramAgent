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
        from phase_agent.analysis.state.project_authority import project_identity

        if (
            decision in {"approve", "confirm_sensitive"}
            and not project_identity(state)["model_consistent"]
        ):
            raise request_error("当前模型版本与文件登记不一致，需先核对；未执行")
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
        from phase_agent.analysis.state.project_authority import project_identity

        proposal = pending.get("agent_proposal") or {}
        current_hash = proposal_hash(proposal)
        if expected_state_version != current_version:
            # summary_id includes read-only facts, token accounting and
            # presentation metadata. Those may change while the immutable
            # pending proposal remains identical. Refresh the UI version in
            # that case; a changed proposal or bound project still fails
            # closed below.
            if expected_proposal_hash != current_hash:
                raise request_error("proposal_hash_mismatch; refresh the approval page")
            bound_identity = pending.get("project_identity")
            if bound_identity and bound_identity != project_identity(state):
                raise request_error("proposal_context_changed; 模型或生效设置已改变，需更新具体方案")
            expected_state_version = current_version

        if (
            decision in {"approve", "confirm_sensitive"}
            and pending.get("project_identity")
            and (pending["project_identity"] != project_identity(state))
        ):
            raise request_error("proposal_context_changed; 模型或生效设置已改变，需更新具体方案")
        if expected_proposal_hash != current_hash:
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

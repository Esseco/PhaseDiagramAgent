"""Versioned approval routing; native persistence is handled by workflow policy."""

from contextlib import ExitStack

from phase_agent.tools.step_runner.file_protocol import read_json
from phase_agent.runtime.review_requests import (
    approval_state_version,
    remember_approval_state_version,
)


def _review_facts(state, pending, plan_id, decision):
    version = approval_state_version(state)
    if decision not in {"approve", "confirm_sensitive"}:
        return version, False
    from phase_agent.policies.recovery_transition import (
        legacy_post_recovery_plan_ids,
        post_recovery_tools,
    )

    allowed = post_recovery_tools(state)
    action = (pending.get("agent_proposal") or {}).get("raw_action") or {}
    blocked = plan_id in legacy_post_recovery_plan_ids(state) or (
        allowed is not None and (action.get("tool") or action.get("action_type")) not in allowed
    )
    return version, blocked


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
    operator="agent",
):
    """Accept a decision only from the authenticated local approval surface."""
    if operator not in {"agent", "codex"}:
        raise ValueError("invalid operator")
    from phase_agent.persistence.state.read_versions import read_identity
    from phase_agent.policies.file_approval import proposal_hash

    with handler.lock, ExitStack() as cleanup:
        from phase_agent.persistence.state.field_index import IndexUnavailable, open_indexed_state
        from phase_agent.runtime.turn_process import has_reusable_initial_state

        state = pending = indexed_facts = None
        # Additional review kinds may write state directly and keep their original
        # full-state, version-checked path.
        if decision in {
            "approve",
            "reject",
            "confirm_sensitive",
        } and not has_reusable_initial_state(handler.state_path):
            indexed = open_indexed_state(handler.state_path)
            if indexed is not None:
                try:
                    pending = (indexed.get("pending_execution_policies") or {}).get(plan_id)
                    if pending is not None:
                        indexed_facts = _review_facts(indexed, pending, plan_id, decision)
                        indexed.verify(strong=True)
                        remember_approval_state_version(indexed, indexed_facts[0])
                        state = indexed
                except (IndexUnavailable, TypeError, AttributeError):
                    indexed_facts = None
        if state is None:
            source_identity = read_identity(handler.state_path, strong=True)
            state = read_json(handler.state_path, {}) or {}
            pending = (state.get("pending_execution_policies") or {}).get(plan_id)
        action = ((pending or {}).get("agent_proposal") or {}).get("raw_action") or {}
        codex = (
            operator == "codex"
            or action.get("decision_source") == "codex"
            or (action.get("_decision_team") or {}).get("backend") == "codex_native"
        )
        if codex and decision == "modify":
            raise request_error("Codex 接管方案的修改请交回 Codex 重新制定；原方案未改变。")
        del action
        if codex:
            from phase_agent.runtime.codex.no_project_model import NoProjectModel

            dependencies = handler.workflow_kwargs
            if "agent_client" in dependencies:
                cleanup.callback(
                    dependencies.__setitem__, "agent_client", dependencies["agent_client"]
                )
            else:
                cleanup.callback(dependencies.pop, "agent_client", None)
            dependencies["agent_client"] = NoProjectModel()

        from phase_agent.evidence.project_authority import project_identity

        if (
            decision in {"approve", "confirm_sensitive"}
            and not project_identity(state)["model_consistent"]
        ):
            raise request_error("当前模型版本与文件登记不一致，需先核对；未执行")
        completed = (state.get("invocations") or {}).get(plan_id) if pending is None else None
        if completed is not None:
            return {"status": "already_processed", "result": completed}
        if pending is None:
            from phase_agent.runtime.review_requests import additional_reviews, review_additional

            rows = additional_reviews(state, handler.workflow_kwargs.get("config_session"))
            row = next((item for item in rows if item["plan_id"] == plan_id), None)
            if row is None:
                raise request_error("plan is not pending")
            version = approval_state_version(state)
            if expected_state_version != version or expected_proposal_hash != row["proposal_hash"]:
                raise request_error("review_changed; refresh the approval page")
            if decision not in {"approve", "reject", "confirm_sensitive", "modify"}:
                raise request_error("invalid approval-page decision")
            return review_additional(
                handler, state, row, decision, comment, expected_identity=source_identity
            )
        current_version, recovery_blocked = indexed_facts or _review_facts(
            state, pending, plan_id, decision
        )
        proposal = pending.get("agent_proposal") or {}
        current_hash = proposal_hash(proposal)
        if recovery_blocked:
            raise request_error(
                "proposal_context_changed; 原方案不属于当前回收后的科学阶段，请重新展示具体方案后审批；未执行"
            )
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
                raise request_error(
                    "proposal_context_changed; 模型或生效设置已改变，需更新具体方案"
                )
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
            del state, pending, proposal
            result = handler._run(
                plan_id, {"decision": "comment", "comment": instruction}, instruction
            )
            return {"status": result.get("status"), "result": result}
        approved = decision in {"approve", "confirm_sensitive"}
        audit_comment = str(comment or "").strip()
        if approved:
            audit_comment = (audit_comment + "\napprove").strip()
        feedback = {"decision": "approve" if approved else "reject", "comment": audit_comment}
        # The graph loads its own verified state and checkpoint. Release the
        # validation snapshot first so large projects do not hold both in memory.
        if indexed_facts is not None:
            state.verify()
            del indexed
        del state, pending, proposal
        result = handler._run(plan_id, feedback, "local approval page")
        return {"status": result.get("status"), "result": result}

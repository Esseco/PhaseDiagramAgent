"""Dialogue nodes; scientific proposals use the bounded decision-team graph."""

from langgraph.runtime import Runtime
from phase_agent.graphs.dialogue.state import TurnRuntime


def read_project(state, runtime: Runtime[TurnRuntime]):
    from phase_agent.tools.step_runner.file_protocol import read_json, write_json
    from phase_agent.runtime.chat_application import (
        _latest_user_message,
        _open_webui_metadata_reply,
        OpenWebUIRequestError,
    )

    context = runtime.context
    handler = context.handler
    message = _latest_user_message(context.messages)
    metadata = _open_webui_metadata_reply(message)
    if metadata is not None:
        return {"reply": metadata, "operation": "done"}
    if handler.conversation_id not in {None, context.conversation_id}:
        raise OpenWebUIRequestError("此运行时已绑定另一个会话。")
    # Early return/review paths must persist bindings for this conversation too.
    handler.conversation_id = context.conversation_id
    missing = object()
    from phase_agent.runtime.turn_process import presentation_signature, remember_presentation_state

    signature = presentation_signature(handler.state_path)
    facts = read_json(handler.state_path, missing)
    if facts is missing:
        from pathlib import Path

        directory = Path(handler.state_path).parent
        existing_project = (directory / "langgraph_lifecycle.sqlite").is_file() or (
            directory / "ledgers" / "phase_data.json"
        ).is_file()
        if handler.config_delegate is None and existing_project:
            raise OpenWebUIRequestError(
                "项目状态文件不存在，无法核对现有任务；本次未分析或生成任务。"
            )
        facts = {}
    facts = facts or {}
    dft_question_reply = None
    if handler.config_delegate is None:
        from phase_agent.runtime.read_only_queries import read_only_reply

        # Queries cannot collect returns or alter the facts being displayed.
        # Approval/rejection still uses the recovery-aware control boundary below.
        answer = read_only_reply(
            handler, message, facts, context.conversation_id, include_review_controls=False
        )
        if answer is not None:
            handler.last_turn_outcome = "answered"
            remember_presentation_state(handler.state_path, facts, signature)
            return {"reply": answer, "operation": "done"}
        from phase_agent.tools.state.dft_recovery_decision import classify_dft_recovery_reply

        dft_question_reply = classify_dft_recovery_reply(message, facts)
        if facts.get("pending_dft_recovery_question"):
            from phase_agent.runtime.review_requests import pending_reviews
            from phase_agent.graphs.dialogue.commands import classify_user_decision

            if pending_reviews(facts, state_path=handler.state_path):
                explicit_dft_reply = classify_dft_recovery_reply(
                    message, facts, allow_short_replies=False
                )
                if explicit_dft_reply is None and (
                    dft_question_reply is not None
                    or classify_user_decision(message) in {"approve", "reject"}
                ):
                    handler.last_turn_outcome = "answered"
                    remember_presentation_state(handler.state_path, facts, signature)
                    return {
                        "reply": (
                            "当前同时存在 DFT 回收问题和待审批方案，请指定回复对象。"
                            "本次未批准、拒绝或执行任何操作。"
                            "DFT 可明确回复‘继续回收’或‘不再回收DFT’；"
                            "方案请在审批页指定具体对象。"
                        ),
                        "operation": "done",
                    }
                dft_question_reply = explicit_dft_reply
    from phase_agent.tools.workflows.compact_action_history import compact_state_history_once

    if compact_state_history_once(facts):
        write_json(handler.state_path, facts)
    from phase_agent.policies.recovery_transition import legacy_post_recovery_plan_ids

    if handler.config_delegate is None:
        pending = facts.get("pending_execution_policies") or {}
        stale_plan_ids = legacy_post_recovery_plan_ids(facts)
        if stale_plan_ids:
            for plan_id in stale_plan_ids:
                for record in facts.get("action_records") or []:
                    if record.get("record_id") == plan_id:
                        record["status"] = "superseded"
                        record["superseded_reason"] = "回收后旧方案未经过当前决策门禁"
            facts["pending_execution_policies"] = {
                key: row for key, row in pending.items() if key not in stale_plan_ids
            }
            facts.setdefault("superseded_pending_plans", []).extend(
                {
                    "plan_id": plan_id,
                    "tool": "legacy_post_recovery_plan",
                    "reason": "legacy_post_recovery_plan_requires_current_decision_boundary",
                }
                for plan_id in stale_plan_ids
            )
            facts["post_recovery_next_step_pending"] = True
            write_json(handler.state_path, facts)
    # An explicit independent run must switch projects before any old-result
    # recovery or semantic model call can inspect the previous state.
    from phase_agent.graphs.dialogue.commands import _classify_history_decision

    if (
        getattr(handler, "history_decision", "continue") is None
        and _classify_history_decision(message) == "new"
    ):
        from phase_agent.graphs.dialogue.history import restore_session

        handler.conversation_id = context.conversation_id
        return {
            "reply": restore_session(handler, message, facts),
            "operation": "done",
        }
    from phase_agent.runtime.turn_process import process_event

    statuses = {row.get("task_id"): row.get("status") for row in facts.get("tasks") or []}
    from phase_agent.runtime.recovery_diagnostics import (
        saved_recovery_error,
        record_recovery_error,
        clear_recovery_error,
    )

    recovery_error = saved_recovery_error(handler.state_path)
    recovery_just_approved = False
    bound_dft_reply = False
    if handler.config_delegate is None:
        from phase_agent.tools.workflows.recover_saved_results import (
            recovery_candidate,
            recover_saved_results,
        )

        recovery = recovery_candidate(facts, state_path=handler.state_path)
        if (
            dft_question_reply is not None
            and not recovery_error
            and not facts.get("pending_execution_recoveries")
            and not (recovery["pending"] and recovery["returned"])
        ):
            from phase_agent.evidence.dft_round_status import DFT_STAGES, dft_round_scope
            from phase_agent.tools.state.execution_receipts import recovery_report

            pending_feedback = set(facts.get("processed_task_ids") or []) - set(
                facts.get("feedback_processed_task_ids") or []
            )
            backlog = [
                row
                for row in facts.get("tasks") or []
                if row.get("task_id") in pending_feedback
                and row.get("stage") in {"relax_and_feature", "deep_search", *DFT_STAGES}
                and row.get("status") in {"completed", "failed", "timeout", "cancelled"}
            ]
            question_scope = facts["pending_dft_recovery_question"]["scope"]
            receipts = recovery_report(handler.state_path, facts)
            bound_dft_reply = (
                receipts["status"] == "clear"
                and not receipts["unsettled"]
                and len(backlog) == recovery["backlog_count"]
                and all(
                    row.get("stage") in DFT_STAGES and dft_round_scope(row) == question_scope
                    for row in backlog
                )
            )
        if not recovery["pending"] and not bound_dft_reply:
            from phase_agent.graphs.dialogue.saved_control import unavailable_review_reply

            unavailable = unavailable_review_reply(handler, message, facts)
            if unavailable is not None:
                handler.last_turn_outcome = "answered"
                return {"reply": unavailable, "operation": "done"}
        normalized_message = message.strip().lower()
        from phase_agent.graphs.dialogue.saved_control import semantic_recovery_intent

        recovery_intent = (
            "question"
            if dft_question_reply is not None
            else semantic_recovery_intent(
                handler, facts, normalized_message, pending=recovery["pending"]
            )
        )
        if recovery["pending"]:
            process_event(
                "回收授权判定",
                {"pending_return": True, "intent": recovery_intent, "scientific_execution": False},
            )
        recovery_approved = recovery_intent == "approve"
        recovery_rejected = recovery_intent == "reject"
        from phase_agent.tools.state.training_pending_validation import training_pending_validation

        training_analysis_requested = recovery_approved and training_pending_validation(facts)
        return_notice = normalized_message.rstrip("。.!！") in {
            "已回收",
            "回收完毕",
            "回收完成",
            "已回传",
            "结果已回传",
            "回传完成",
        }
        if return_notice and not recovery["pending"]:
            from phase_agent.runtime.review_requests import pending_reviews

            if pending_reviews(facts):
                from phase_agent.runtime.plan_queries import pending_plan_reply
                from phase_agent.graphs.dialogue.support import bind_presented_proposal

                bind_presented_proposal(handler, facts)
                handler.last_turn_outcome = "answered"
                return {"reply": pending_plan_reply(facts), "operation": "done"}
        if (
            recovery_approved
            and not recovery["pending"]
            and not training_analysis_requested
            and not return_notice
        ):
            handler.last_turn_outcome = "answered"
            return {
                "reply": "当前没有待回收的未核对结果；本轮未批准或执行任何科学方案。",
                "operation": "done",
            }
        if recovery["pending"] and not recovery_approved and not bound_dft_reply:
            if recovery_rejected:
                handler.last_turn_outcome = "answered"
                return {
                    "reply": (
                        "已暂不回收当前返回结果；本轮未读取、未入账、未分析。"
                        "如需回收，请明确回复‘同意回收’。"
                    ),
                    "operation": "done",
                }
            handler.last_turn_outcome = "answered"
            from phase_agent.graphs.dialogue.saved_control import recovery_confirmation_reply

            return {
                "reply": recovery_confirmation_reply(facts),
                "operation": "done",
            }
        try:
            # Existing question choices are administrative, not consent to read
            # new returns. Their maintenance handler retains the bound scope.
            recovered = (
                facts
                if bound_dft_reply
                else recover_saved_results(facts, **handler.workflow_kwargs)
            )
            report = recovered.get("result_collection") or {}
            if report.get("recovery_error"):
                raise ValueError(report["recovery_error"])
            # A newly completed collection report proves this attempt ran.
            # Object identity alone cannot distinguish a valid return from
            # a collector's zero-result validation/transfer failure.
            if (
                recovery["pending"]
                and recovered is not facts
                and report
                and report is not facts.get("result_collection")
            ):
                clear_recovery_error(handler.state_path)
                recovery_error = None
            facts = recovered
        except Exception as error:
            # Recovery may have committed a subset of reconciled facts before
            # discovering an invalid return. Reload that committed snapshot so
            # the next answer never reasons from stale in-memory state.
            committed = read_json(handler.state_path, missing)
            if committed is missing or not isinstance(committed, dict):
                raise
            facts = committed or {}
            recovery_error = str(error)[:500] or error.__class__.__name__
            record_recovery_error(handler.state_path, recovery_error, recovery.get("fingerprint"))
            process_event(
                "结果回收失败，进入只读答疑",
                {"error": recovery_error, "scientific_execution": False},
            )
        if (
            not recovery_error
            and recovery_approved
            and (recovery["pending"] or training_analysis_requested)
        ):
            # Recovery approval authorizes observation only.  Once observation
            # succeeds, immediately continue through the normal planning path
            # so the next action is proposed and held for a separate approval.
            # Any older unapproved plan was based on pre-recovery facts and must
            # not become an implicit approval or hide the new scientific choice.
            pending = facts.get("pending_execution_policies") or {}
            facts["recovery_acknowledgement"] = {
                "fingerprint": recovery.get("fingerprint"),
                "scope": "explicit_user_recovery_approval",
                "source_stages": sorted(
                    {
                        row["stage"]
                        for row in facts.get("tasks") or []
                        if row.get("stage")
                        and row.get("status") in {"completed", "failed", "timeout", "cancelled"}
                        and row.get("status") != statuses.get(row.get("task_id"))
                    }
                ),
            }
            # This marker survives a reload.  It is consumed only after a
            # next-step proposal is approved, so an old Relax/MC proposal
            # cannot become the next action by saved-plan routing.
            facts["post_recovery_next_step_pending"] = True
            if pending:
                superseded = []
                for plan_id, row in pending.items():
                    action = (row.get("agent_proposal") or {}).get("raw_action") or {}
                    superseded.append(
                        {
                            "plan_id": plan_id,
                            "tool": action.get("tool"),
                            "reason": "result_recovery_completed_before_approval",
                        }
                    )
                    for record in facts.get("action_records") or []:
                        if record.get("record_id") == plan_id and record.get("status") in {
                            None,
                            "awaiting_approval",
                            "prepared",
                        }:
                            record["status"] = "superseded"
                            record["superseded_reason"] = "结果回收完成后按新证据重新提出下一步方案"
                facts["pending_execution_policies"] = {}
                facts.setdefault("superseded_pending_plans", []).extend(superseded)
            write_json(handler.state_path, facts)
            facts["_recovery_just_approved"] = True
            recovery_just_approved = True
            message = "继续"

    # A continuation while registered work is still pending is a read-only
    # wait query.  Resolve it before the dialogue/lifecycle checkpoint path so
    # an old checkpoint cannot turn the same current state into a recovery
    # block or a new model proposal.
    normalized_message = message.strip().lower()
    if (
        handler.config_delegate is None
        and normalized_message in {"继续", "continue", "继续原运行"}
        and not recovery_error
        and not recovery_just_approved
        and not facts.get("pending_dft_recovery_question")
        and not facts.get("pending_execution_recoveries")
        and not facts.get("pending_execution_policies")
    ):
        from phase_agent.tools.state.execution_receipts import recovery_report

        current_receipts = recovery_report(handler.state_path, facts)
        if current_receipts.get("status") == "clear" and not current_receipts.get("unsettled"):
            from phase_agent.runtime.review_requests import pending_reviews

            if not pending_reviews(facts):
                from phase_agent.graphs.dialogue.saved_control import active_task_wait_reply

                waiting = active_task_wait_reply(facts)
                if waiting:
                    handler.last_turn_outcome = "answered"
                    remember_presentation_state(handler.state_path, facts, signature)
                    return {"reply": waiting, "operation": "done"}

    from phase_agent.runtime.read_only_queries import read_only_reply
    from phase_agent.runtime.turn_process import timed_call

    answer = timed_call(
        "read_only_query",
        read_only_reply,
        handler,
        message,
        facts,
        context.conversation_id,
        include_review_controls=not bound_dft_reply,
    )
    if answer is not None:
        handler.last_turn_outcome = "answered"
        remember_presentation_state(handler.state_path, facts, signature)
        return {"reply": answer, "operation": "done"}
    recovered = {}
    for row in facts.get("tasks") or []:
        if row.get("status") in {"completed", "failed", "timeout", "cancelled"} and row.get(
            "status"
        ) != statuses.get(row.get("task_id")):
            counts = recovered.setdefault(
                row.get("stage") or "unknown", {"completed": 0, "failed": 0}
            )
            counts["completed" if row["status"] == "completed" else "failed"] += 1
    return {
        "message": message,
        "facts": facts,
        "verified_recovery": recovered,
        "recovery_error": recovery_error,
        "recovery_just_approved": recovery_just_approved,
        "operation": "dispatch",
    }


def history(state, runtime: Runtime[TurnRuntime]):
    from phase_agent.graphs.dialogue.history import restore_session

    context = runtime.context
    if context.handler.config_delegate is not None:
        return {}
    context.handler.conversation_id = context.conversation_id
    prior = context.handler.history_decision
    reply = restore_session(context.handler, state["message"], state["facts"])
    return {"reply": reply} if reply is not None else {"history_restored": prior is None}


def route_request(state, runtime: Runtime[TurnRuntime]):
    handler = runtime.context.handler
    handler.conversation_id = runtime.context.conversation_id
    if state.get("recovery_error"):
        # Do not let saved controls or maintenance nodes mutate a project while
        # its result reconciliation is known to be incomplete.
        return {"operation": "decide"}
    command = state["message"].strip().lower()
    from phase_agent.runtime.review_requests import pending_reviews

    reviews = pending_reviews(state["facts"])
    from phase_agent.tools.workflows.model_refresh_state import current_model_refresh

    if (
        not reviews
        and command in {"同意", "approve"}
        and current_model_refresh(state["facts"]).get("status")
        in {"approval_required", "supplement_ready"}
    ):
        from phase_agent.graphs.invocation_context import new_invocation

        runtime.context.scientific_call = lambda: handler._run(
            new_invocation(),
            None,
            "当前新模型刷新阶段缺少已保存的待审方案。请重新展示生成刷新文件的具体范围和预算并等待批准；"
            "本次同意没有对应的有效方案，不作为新方案的批准，不生成输入或提交计算。",
        )
        return {"operation": "scientific_workflow"}
    # Resolve references to saved work before invoking scientific planning.
    from phase_agent.graphs.dialogue.saved_control import route_saved_control
    from phase_agent.tools.step_runner.file_protocol import write_json

    # A result-recovery approval starts a fresh scientific decision.  Saved
    # Relax/MC input plans were created from pre-recovery facts and must not
    # be shown or reused on the next turn.  Keep a valid DFT/branch proposal
    # available for its normal approval path.
    post_recovery_gate = bool(
        state["facts"].get("post_recovery_next_step_pending")
        or state["facts"].get("_recovery_just_approved")
    )
    if post_recovery_gate:
        pending = state["facts"].get("pending_execution_policies") or {}
        from phase_agent.policies.recovery_transition import post_recovery_tools

        allowed_tools = post_recovery_tools(state["facts"])
        stale_ids = []
        for plan_id, row in pending.items():
            action = (row.get("agent_proposal") or {}).get("raw_action") or {}
            if (action.get("tool") or action.get("action_type")) not in allowed_tools:
                stale_ids.append(plan_id)
        if stale_ids:
            superseded = []
            for plan_id in stale_ids:
                row = pending.pop(plan_id, {})
                action = (row.get("agent_proposal") or {}).get("raw_action") or {}
                superseded.append(
                    {
                        "plan_id": plan_id,
                        "tool": action.get("tool") or action.get("action_type"),
                        "reason": "post_recovery_next_step_requires_stage_decision",
                    }
                )
                for record in state["facts"].get("action_records") or []:
                    if record.get("record_id") == plan_id and record.get("status") in {
                        None,
                        "awaiting_approval",
                        "prepared",
                    }:
                        record["status"] = "superseded"
                        record["superseded_reason"] = "回收后下一步须按本轮结果重新提出科学方案"
            state["facts"]["pending_execution_policies"] = pending
            state["facts"].setdefault("superseded_pending_plans", []).extend(superseded)
            write_json(handler.state_path, state["facts"])
        if stale_ids or not pending:
            return {"operation": "decide"}

    routed = route_saved_control(state, runtime, reviews)
    if routed is not None:
        return routed
    if (
        handler.config_delegate is None
        and not reviews
        and command in {"继续", "continue", "继续原运行"}
    ):
        from phase_agent.graphs.dialogue.saved_control import active_task_wait_reply

        waiting = active_task_wait_reply(state["facts"])
        if waiting is not None:
            handler.last_turn_outcome = "answered"
            return {"reply": waiting}
    if (
        handler.config_delegate is None
        and not reviews
        and command in {"同意", "approve", "继续", "continue", "继续原运行"}
    ):
        from phase_agent.tools.state.execution_receipts import recovery_report

        if recovery_report(handler.state_path, state["facts"])["unsettled"]:
            from phase_agent.graphs.dialogue.saved_control import recovery_reply

            return {"reply": recovery_reply(handler)}
        from phase_agent.tools.workflows.reuse_dft_selection import reusable_selection

        if reusable_selection(state["facts"]) is not None:
            from phase_agent.graphs.dialogue.saved_selection_resume import prepare_resume

            prepare_resume(handler, runtime.context)
            return {"operation": "scientific_workflow"}
    from phase_agent.graphs.dialogue.saved_control import unavailable_review_reply

    unavailable = unavailable_review_reply(handler, state["message"], state["facts"])
    if unavailable is not None:
        handler.last_turn_outcome = "answered"
        return {"reply": unavailable}
    # Existing explicit control messages do not request a revised scientific
    # proposal. Reuse the saved approval owner instead of asking an LLM again.
    if (
        handler.config_delegate is None
        and reviews
        and command in {"继续", "continue", "继续原运行"}
    ):
        from phase_agent.runtime.plan_queries import pending_plan_reply
        from phase_agent.graphs.dialogue.support import bind_presented_proposal

        pending = state["facts"].get("pending_execution_policies") or {}
        if len(pending) == 1:
            from phase_agent.tools.state.execution_receipts import recovery_report

            if recovery_report(handler.state_path, state["facts"])["unsettled"]:
                return {"reply": "上次执行仍需核对，请在审批页查看恢复证据；本次未重新分析或执行。"}
            from phase_agent.runtime.workflow_reply_presentation import _is_model_failure_proposal

            invocation, stored = next(iter(pending.items()))
            if _is_model_failure_proposal(stored.get("agent_proposal") or {}):
                runtime.context.scientific_call = lambda: handler._run(
                    invocation,
                    {"decision": "comment", "comment": state["message"]},
                    state["message"],
                )
                return {"operation": "scientific_workflow"}
            from phase_agent.graphs.approval_graph import durable_approval
            from phase_agent.policies.file_approval import proposal_hash

            approval = durable_approval(
                handler.state_path,
                {
                    "invocation_id": invocation,
                    "config_version": state["facts"].get("confirmed_config_version"),
                    "proposal_hash": proposal_hash(stored.get("agent_proposal") or {}),
                    "revision": stored.get("revision", 0),
                },
            )
            if approval["status"] == "approval_reconciliation_required":
                return {
                    "reply": "上次审批决定已交付，业务记录尚需核对；本次未重新批准、分析或执行。"
                }
            from phase_agent.graphs.dialogue.saved_control import (
                generation_blocked_review,
                prepare_saved_revision,
            )

            blocked = generation_blocked_review(state["facts"], reviews)
            if blocked is not None:
                return prepare_saved_revision(handler, runtime.context, blocked, state["message"])
        bind_presented_proposal(handler, state["facts"])
        return {"reply": pending_plan_reply(state["facts"])}
    if command in {"确认敏感操作", "confirm sensitive action"}:
        import os

        return {
            "reply": "请在本机审批页核对具体方案后批准："
            + f"http://127.0.0.1:{os.environ.get('PHASE_CONTROL_PORT', '8765')}/phase/approval"
        }
    if handler.config_delegate is not None:
        operation = "configure"
    elif command in {"approve", "同意", "reject", "拒绝"} and reviews:
        operation = "review"
    elif command in {"配置修订", "修改配置"}:
        operation = "configure"
    else:
        operation = "decide"
    return {"operation": operation}


def configure(state, runtime: Runtime[TurnRuntime]):
    context = runtime.context
    handler = context.handler
    if handler.config_delegate is None:
        if not callable(handler.config_revision_factory):
            return {"reply": "配置修订入口未配置，当前配置和待审批方案未变。"}
        handler.config_delegate = handler.config_revision_factory(state["facts"])
    return {
        "reply": handler.config_delegate(context.messages, conversation_id=context.conversation_id)
    }


def inspect_configuration(state, runtime: Runtime[TurnRuntime]):
    from phase_agent.graphs.dialogue.configuration_view import configuration_view

    runtime.context.handler.last_turn_outcome = "configuration_view_requested"
    reply = configuration_view(runtime.context.handler, state["facts"])
    if state.get("recovery_error"):
        from phase_agent.graphs.dialogue.decision import recovery_notice

        reply = recovery_notice(state["recovery_error"]) + "\n" + reply
    return {"reply": reply}


def review(state, runtime: Runtime[TurnRuntime]):
    from phase_agent.graphs.dialogue.support import review_presented_proposal

    return {
        "reply": review_presented_proposal(
            runtime.context.handler,
            state["facts"],
            state["message"],
            requested_review=(state.get("result") or {}).get("review"),
        )
    }


def decide(state, runtime: Runtime[TurnRuntime]):
    from phase_agent.graphs.invocation_context import new_invocation
    from phase_agent.graphs.dialogue.support import fresh_turn_context

    handler = runtime.context.handler
    pending = state["facts"].get("pending_execution_policies") or {}
    invocation = next(iter(pending)) if pending else new_invocation()
    feedback = (
        {"decision": "comment", "comment": state["message"]}
        if pending and not state.get("history_restored")
        else None
    )
    handler.turn_context = fresh_turn_context(
        handler,
        state["facts"],
        runtime.context.messages,
        runtime.context.conversation_id,
    )
    from phase_agent.graphs.invocation_context import dialogue_message

    from phase_agent.graphs.dialogue.decision import decide_from_saved_facts

    def request_scientific(client=None):
        def execute():
            previous = handler.workflow_kwargs.get("agent_client")
            if client is not None:
                handler.workflow_kwargs["agent_client"] = client
            try:
                with dialogue_message():
                    return handler._run(invocation, feedback, state["message"])
            finally:
                if client is not None:
                    handler.workflow_kwargs["agent_client"] = previous

        runtime.context.scientific_call = execute
        return {"status": "scientific_requested"}

    if state.get("recovery_just_approved") and not state.get("recovery_error"):
        # Recovery approval permits this observation, followed by a new proposal;
        # it cannot approve that proposal or execute the next scientific action.
        from phase_agent.runtime.turn_process import process_event

        process_event("回收后进入下一步规划", {"new_action_approval": False})
        request_scientific()
        return {"operation": "scientific_workflow"}

    # Known continuation enters observation/planning directly, not semantic routing.
    if (
        state["message"].strip().lower().rstrip("。.!！")
        in {
            "继续",
            "continue",
            "继续原运行",
            "已回收",
            "回收完毕",
            "回收完成",
            "回传完成",
            "结果已回传",
            "已回传",
        }
        and not pending
        and not state.get("recovery_error")
        and not state.get("recovery_just_approved")
    ):
        request_scientific()
        return {"operation": "scientific_workflow"}
    with dialogue_message():
        result = decide_from_saved_facts(
            handler,
            state["facts"],
            state["message"],
            request_scientific,
            defer_scientific=request_scientific,
            evidence=state.get("evidence") or {},
            recovery_error=state.get("recovery_error"),
        )
    return {
        "result": result,
        "facts": result.get("state") or state["facts"],
        "operation": {
            "scientific_requested": "scientific_workflow",
            "configuration_edit_requested": "configure",
            "configuration_view_requested": "inspect_configuration",
            "proposal_review_requested": "review",
            "evidence_requested": "read_evidence",
        }.get(result.get("status"), "present"),
    }


def read_evidence(state, runtime: Runtime[TurnRuntime]):
    from phase_agent.graphs.dialogue.evidence import read_topics, read_limit_reply
    from phase_agent.runtime.turn_process import process_event

    topics = state["result"]["topics"]
    prior = set(state.get("evidence_topics") or [])
    if int(state.get("evidence_reads", 0)) >= 4 or set(topics).issubset(prior):
        process_event(
            "停止重复读取", {"topics": topics, "read_count": state.get("evidence_reads", 0)}
        )
        return {
            "result": {
                "status": "answered",
                "answer": read_limit_reply(state["facts"]),
                "state": state["facts"],
                "submitted": False,
                "steps_executed": 0,
            },
            "operation": "present",
        }
    new = [topic for topic in topics if topic not in prior]
    if int(state.get("evidence_reads", 0)) == 3:
        # Complex conversation gets all remaining evidence before its final answer.
        new = [
            topic
            for topic in (
                "mc",
                "dft",
                "tasks",
                "model",
                "convergence",
                "budget",
                "history",
                "configuration",
                "plans",
            )
            if topic not in prior
        ]
    result = read_topics(
        state["facts"], new, manager=runtime.context.handler.workflow_kwargs.get("manager")
    )
    process_event("按需读取证据", {"topics": new, "read_only": True})
    return {
        "evidence": {
            **(state.get("evidence") or {}),
            **result,
            "_loaded_topics": sorted(prior | set(new)),
        },
        "evidence_topics": sorted(prior | set(new)),
        "evidence_reads": int(state.get("evidence_reads", 0)) + 1,
        "operation": "decide",
    }


def scientific_workflow(state, runtime: Runtime[TurnRuntime]):
    """Execute the prepared invocation; proposal reuse stays in Runtime."""
    execute = runtime.context.scientific_call
    if not callable(execute):
        raise RuntimeError("Scientific invocation was not prepared by the decision node")
    return {"result": execute(), "operation": "present"}


def present(state, runtime: Runtime[TurnRuntime]):
    from phase_agent.graphs.dialogue.support import bind_presented_proposal
    from phase_agent.tools.step_runner.file_protocol import read_json
    from phase_agent.runtime.workflow_reply_presentation import format_workflow_reply

    handler = runtime.context.handler
    handler.last_turn_outcome = state["result"].get("status")
    if state["result"].get("status") in {"awaiting_approval", "training_handoff"}:
        bind_presented_proposal(handler, read_json(handler.state_path, {}) or {})
    result = {**state["result"], "verified_recovery": state.get("verified_recovery") or {}}
    return {"reply": format_workflow_reply(result, handler.state_path)}


def operation_node(handle):
    def node(state, runtime: Runtime[TurnRuntime]):
        context = runtime.context
        facts = state["facts"]
        if state.get("recovery_error"):
            return {"facts": facts}
        reply = handle(
            context.handler,
            state["message"],
            facts,
            context.messages,
            context.conversation_id,
        )
        return {"facts": facts, **({"reply": reply} if reply is not None else {})}

    return node

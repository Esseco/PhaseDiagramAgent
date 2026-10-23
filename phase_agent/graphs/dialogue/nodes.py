"""Dialogue graph nodes; scientific decisions delegate to one model call."""

from langgraph.runtime import Runtime
from phase_agent.graphs.dialogue.state import TurnRuntime


def read_project(state, runtime: Runtime[TurnRuntime]):
    from phase_agent.tools.step_runner.file_protocol import read_json
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
    from phase_agent.runtime.read_only_queries import read_only_reply
    from phase_agent.runtime.turn_process import timed_call

    answer = timed_call(
        "read_only_query", read_only_reply, handler, message, facts, context.conversation_id
    )
    if answer is not None:
        handler.last_turn_outcome = "answered"
        remember_presentation_state(handler.state_path, facts, signature)
        return {"reply": answer, "operation": "done"}
    statuses = {row.get("task_id"): row.get("status") for row in facts.get("tasks") or []}
    if handler.config_delegate is None:
        from phase_agent.tools.workflows.recover_saved_results import (
            recover_saved_results,
        )

        facts = recover_saved_results(facts, **handler.workflow_kwargs)
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
    command = state["message"].strip().lower()
    from phase_agent.runtime.review_requests import pending_reviews

    reviews = pending_reviews(state["facts"])
    # Resolve references to saved work before invoking scientific planning.
    from phase_agent.graphs.dialogue.saved_control import route_saved_control

    routed = route_saved_control(state, runtime, reviews)
    if routed is not None:
        return routed
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
            from phase_agent.graphs.approval_graph import durable_approval
            from phase_agent.tools.policy.file_approval import proposal_hash

            invocation, stored = next(iter(pending.items()))
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
    return {"reply": configuration_view(runtime.context.handler, state["facts"])}


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

    # Known continuation enters observation/planning directly, not semantic routing.
    if (
        state["message"].strip().lower().rstrip("。.!！")
        in {
            "继续",
            "continue",
            "继续原运行",
            "回收完毕",
            "回收完成",
            "回传完成",
            "结果已回传",
            "已回传",
        }
        and not pending
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
    if state["result"].get("status") == "awaiting_approval":
        bind_presented_proposal(handler, read_json(handler.state_path, {}) or {})
    result = {**state["result"], "verified_recovery": state.get("verified_recovery") or {}}
    return {"reply": format_workflow_reply(result, handler.state_path)}


def operation_node(handle):
    def node(state, runtime: Runtime[TurnRuntime]):
        context = runtime.context
        facts = state["facts"]
        reply = handle(
            context.handler,
            state["message"],
            facts,
            context.messages,
            context.conversation_id,
        )
        return {"facts": facts, **({"reply": reply} if reply is not None else {})}

    return node

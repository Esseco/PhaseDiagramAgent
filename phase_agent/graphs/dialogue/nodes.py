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
    facts = read_json(handler.state_path, {}) or {}
    return {"message": message, "facts": facts, "operation": "dispatch"}


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
    if command in {"确认敏感操作", "confirm sensitive action"}:
        import os

        return {
            "reply": "请在本机审批页核对具体方案后批准："
            + f"http://127.0.0.1:{os.environ.get('PHASE_CONTROL_PORT', '8765')}/phase/approval"
        }
    if handler.config_delegate is not None:
        operation = "configure"
    elif command in {"approve", "同意", "reject", "拒绝"} and state["facts"].get(
        "pending_execution_policies"
    ):
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
            runtime.context.handler, state["facts"], state["message"]
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
        handler, state["facts"], runtime.context.messages, runtime.context.conversation_id
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

    with dialogue_message():
        result = decide_from_saved_facts(
            handler,
            state["facts"],
            state["message"],
            request_scientific,
            defer_scientific=request_scientific,
        )
    return {
        "result": result,
        "operation": {
            "scientific_requested": "scientific_workflow",
            "configuration_edit_requested": "configure",
            "configuration_view_requested": "inspect_configuration",
        }.get(result.get("status"), "present"),
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
    return {"reply": format_workflow_reply(state["result"], handler.state_path)}


def operation_node(handle):
    def node(state, runtime: Runtime[TurnRuntime]):
        from phase_agent.tools.step_runner.file_protocol import read_json

        context = runtime.context
        facts = read_json(context.handler.state_path, {}) or {}
        reply = handle(
            context.handler, state["message"], facts, context.messages, context.conversation_id
        )
        return {"facts": facts, **({"reply": reply} if reply is not None else {})}

    return node

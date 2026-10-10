"""Studio entry into the same-process project handler, with replay protection."""

import hashlib
import os
from pathlib import Path
from typing import Annotated, TypedDict, Literal
from langchain_core.messages import AnyMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime


class StudioContext(TypedDict, total=False):
    response_detail: Literal["brief", "detailed"]


class StudioState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    request_text: str
    request_thread: str
    reply: str | None


class StudioChatInput(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class StudioChatOutput(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def user_text(messages):
    """Accept text blocks without discarding attachments or replaying old turns."""
    if not messages or messages[-1].type != "human":
        raise ValueError("请在 messages 中添加一条新的用户消息，再运行；不要重跑旧的助手消息。")
    content = messages[-1].content
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ):
                parts.append(block["text"])
            else:
                raise ValueError("当前入口只接受文字；请移除图片、文件或其他非文本内容后重试。")
        text = "\n".join(parts)
    else:
        raise ValueError("无法识别用户消息格式，请输入文字后重试。")
    if not text.strip():
        raise ValueError("消息为空，请输入指令后再运行。")
    return text


def send_local_message(text, thread_id):
    from phase_agent.runtime.studio_runtime import send_project_message

    return send_project_message(text, thread_id)


def build_studio_graph(sender=send_local_message, *, receipt_path=None, checkpointer=None):
    from phase_agent.graphs.scientific_graph import scientific_graph, use_scientific_graph

    science = scientific_graph()

    def receive_message(state, config):
        messages = state.get("messages") or []
        try:
            text = user_text(messages)
        except ValueError as error:
            return {"reply": f"未执行：{error}", "request_text": "", "request_thread": ""}
        message = messages[-1]
        thread = (config.get("configurable") or {}).get("thread_id")
        if not thread or not message.id:
            raise ValueError("Persistent thread_id and message ID are required")
        return {"request_text": text, "request_thread": thread, "reply": None}

    def agent(state, config, runtime: Runtime[StudioContext]):
        messages = state["messages"]
        message = messages[-1]
        text = state["request_text"]
        thread = state["request_thread"]
        digest = hashlib.sha256(text.encode()).hexdigest()
        path = receipt_path or os.environ.get("PHASE_STUDIO_RECEIPT_PATH")
        if not path:
            raise ValueError(
                "PHASE_STUDIO_RECEIPT_PATH must point to dedicated Studio gateway state"
            )
        from phase_agent.tools.state.execution_receipts import (
            begin_execution,
            record_execution_return,
        )

        execution_identity = {
            "invocation_id": f"{thread}:{message.id}",
            "config_version": "studio-gateway-v1",
            "action_hash": digest,
            "tool": "studio_send_message",
        }
        claim = begin_execution(Path(path), execution_identity)
        from phase_agent.runtime.studio_recovery import can_resume_message

        recoverable = (
            not claim["allowed"]
            and claim.get("reason") != "identity_changed"
            and claim.get("phase") == "started"
            and can_resume_message(
                os.environ.get("PHASE_AGENT_RUNTIME_CONFIG"), execution_identity["invocation_id"]
            )
        )
        if not claim["allowed"] and not recoverable:
            return {
                "messages": [AIMessage(content="该消息已尝试发送，禁止历史重放；请核对业务状态。")],
                "reply": None,
            }
        # Captured compiled child is statically discoverable by LangGraph.
        # The existing command router decides whether this message invokes it.
        from phase_agent.runtime.response_preferences import response_detail

        detail = (runtime.context or {}).get("response_detail", "brief")
        from phase_agent.graphs.invocation_context import scientific_message
        from phase_agent.runtime.studio_session_scope import studio_project_session

        history = [
            {"role": "user" if item.type == "human" else "assistant", "content": item.content}
            for item in messages
            if item.type in {"human", "ai"}
        ]
        with (
            use_scientific_graph(science),
            response_detail(detail),
            scientific_message(execution_identity["invocation_id"]),
            studio_project_session(history),
        ):
            from phase_agent.graphs.cancellation import check_cancelled

            check_cancelled()
            reply = sender(text, thread)
            check_cancelled()
        record_execution_return(path, execution_identity, "completed")
        return {"messages": [AIMessage(content=reply)], "reply": None}

    async def async_agent(state, config, runtime: Runtime[StudioContext]):
        from phase_agent.graphs.cancellation import cancellable_worker

        return await cancellable_worker(agent, state, config, runtime)

    graph = StateGraph(
        StudioState,
        input_schema=StudioChatInput,
        output_schema=StudioChatOutput,
        context_schema=StudioContext,
    )
    from langchain_core.runnables import RunnableLambda
    from langgraph.runtime import get_runtime

    def sync_entry(state, config):
        return agent(state, config, get_runtime(StudioContext))

    async def async_entry(state, config):
        return await async_agent(state, config, get_runtime(StudioContext))

    def respond(state):
        # Successful agent turns already emitted the sole assistant message.
        return {"messages": [AIMessage(content=state["reply"])]} if state.get("reply") else {}

    graph.add_node("receive_message", receive_message)
    graph.add_node("agent", RunnableLambda(sync_entry, afunc=async_entry))
    graph.add_node("respond", respond)
    graph.add_edge(START, "receive_message")
    graph.add_conditional_edges(
        "receive_message",
        lambda state: "respond" if state.get("reply") else "agent",
        ["respond", "agent"],
    )
    graph.add_edge("agent", "respond")
    graph.add_edge("respond", END)
    from phase_agent.graphs.subgraph_visibility import expose_child
    from phase_agent.graphs.dialogue.graph import build_project_turn_graph

    return expose_child(
        graph.compile(checkpointer=checkpointer, name="phase_agent_workflow"),
        "agent",
        build_project_turn_graph(science),
    )


graph = build_studio_graph()

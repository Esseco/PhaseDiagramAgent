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


class StudioState(TypedDict):
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
            elif (isinstance(block, dict) and block.get("type") == "text"
                  and isinstance(block.get("text"), str)):
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
    from run.studio_runtime import send_project_message
    return send_project_message(text, thread_id)


def build_studio_graph(sender=send_local_message, *, receipt_path=None, checkpointer=None):
    from orchestration.scientific_graph import scientific_graph, use_scientific_graph
    science = scientific_graph()
    def chat(state, config, runtime: Runtime[StudioContext]):
        messages = state.get("messages") or []
        try:
            text = user_text(messages)
        except ValueError as error:
            return {"messages": [AIMessage(content=f"未执行：{error}")]}
        message = messages[-1]
        thread = (config.get("configurable") or {}).get("thread_id")
        if not thread or not message.id:
            raise ValueError("Persistent thread_id and message ID are required")
        digest = hashlib.sha256(text.encode()).hexdigest()
        path = receipt_path or os.environ.get("PHASE_STUDIO_RECEIPT_PATH")
        if not path:
            raise ValueError("PHASE_STUDIO_RECEIPT_PATH must point to dedicated Studio gateway state")
        from execution_layer.state.execution_receipts import begin_execution, record_execution_return
        execution_identity = {"invocation_id": f"{thread}:{message.id}", "config_version": "studio-gateway-v1",
                              "action_hash": digest, "tool": "studio_send_message"}
        claim = begin_execution(Path(path), execution_identity)
        if not claim["allowed"]:
            return {"messages": [AIMessage(content="该消息已尝试发送，禁止历史重放；请核对业务状态。") ]}
        # Captured compiled child is statically discoverable by LangGraph.
        # The existing command router decides whether this message invokes it.
        from run.response_preferences import response_detail
        detail = (runtime.context or {}).get("response_detail", "brief")
        with use_scientific_graph(science), response_detail(detail):
            reply = sender(text, thread)
        record_execution_return(path, execution_identity, "completed")
        return {"messages": [AIMessage(content=reply)]}

    graph = StateGraph(StudioState, context_schema=StudioContext)
    graph.add_node("confirmed_local_chat", chat)
    graph.add_edge(START, "confirmed_local_chat")
    graph.add_edge("confirmed_local_chat", END)
    return graph.compile(checkpointer=checkpointer)


graph = build_studio_graph()

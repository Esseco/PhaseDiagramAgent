"""Agent Server owns one project handler shared by Studio and control pages."""

from contextlib import asynccontextmanager
import os
from pathlib import Path
import threading
from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.routing import Route
from phase_agent.runtime.studio_conversation_panel import PAGE

_handler = None
_message_lock = threading.RLock()


def project_handler(owner):
    """Follow only stored delegate references, never fabricate dynamic attributes."""
    from phase_agent.runtime.chat_application import RunWorkflowChatHandler

    seen = set()
    while owner is not None and id(owner) not in seen:
        if isinstance(owner, RunWorkflowChatHandler):
            return owner
        seen.add(id(owner))
        owner = getattr(owner, "__dict__", {}).get("delegate")
    return None


def send_project_message(text, thread_id):
    if _handler is None:
        raise RuntimeError("Studio project runtime has not started")
    from phase_agent.runtime.agent_api import _chat_content

    with _message_lock:
        from phase_agent.graphs.cancellation import check_cancelled

        check_cancelled()
        from phase_agent.runtime.turn_process import record_turn, process_event

        with record_turn(os.environ["PHASE_AGENT_RUNTIME_CONFIG"], text, thread_id):
            from phase_agent.runtime.studio_session_scope import studio_project_session

            with studio_project_session():
                reply = _chat_content(
                    _handler([{"role": "user", "content": text}], conversation_id=thread_id)
                )
            process_event("完整答复", {"reply": reply})
            from phase_agent.runtime.response_preferences import detailed_response

            if not detailed_response():
                from phase_agent.runtime.studio_reply_presentation import format_turn_reply
                from phase_agent.runtime.chat_application import RunWorkflowChatHandler
                from phase_agent.tools.step_runner.file_protocol import read_json

                handler = project_handler(_handler)
                if handler is not None:
                    state = read_json(handler.state_path, {}) or {}
                    reply = format_turn_reply(reply, state, configuring=False)
                else:
                    from phase_agent.runtime.configuration_chat import ConfigurationChatHandler

                    if isinstance(_handler, ConfigurationChatHandler):
                        reply = format_turn_reply(reply, {}, configuring=True)
            # Always show the factual epoch/stage, including detailed replies.
            from phase_agent.runtime.chat_state_presentation import format_epoch_reply
            from phase_agent.tools.step_runner.file_protocol import read_json

            current_handler = project_handler(_handler)
            current_state = (
                read_json(current_handler.state_path, {}) or {}
                if current_handler is not None
                else {}
            )
            reply = format_epoch_reply(reply, current_state, configuring=current_handler is None)
            process_event("本轮答复", {"reply": reply})
            return reply


@asynccontextmanager
async def lifespan(app):
    global _handler
    from phase_agent.runtime.agent_api import create_agent_runtime
    from phase_agent.runtime.agent_web_service import create_control_server

    config = os.environ.get("PHASE_AGENT_RUNTIME_CONFIG")
    if not config or not Path(config).is_file():
        raise RuntimeError(
            "PHASE_AGENT_RUNTIME_CONFIG must identify an existing project configuration"
        )
    if _handler is not None:
        raise RuntimeError("Studio runtime already owns a project handler")
    from phase_agent.runtime.studio_run_lifecycle import recover_studio_runs

    await recover_studio_runs()
    handler = create_agent_runtime(config)
    control = create_control_server(
        handler, config, port=int(os.environ.get("PHASE_CONTROL_PORT", "8765"))
    )
    worker = threading.Thread(target=control.serve_forever, daemon=True)
    _handler = handler
    worker.start()
    try:
        yield
    finally:
        _handler = None
        control.shutdown()
        control.server_close()
        worker.join(timeout=5)


async def flow_panel(request):
    from phase_agent.runtime.chat_application import RunWorkflowChatHandler
    from phase_agent.runtime.flow_panel import render_flow_panel
    from phase_agent.tools.step_runner.file_protocol import read_json

    handler = project_handler(_handler)
    if handler is None:
        return HTMLResponse("配置尚未确认，请先在 Studio 完成配置。", status_code=503)
    with handler.lock:
        state = read_json(handler.state_path, {}) or {}
        content = render_flow_panel(state)
    return HTMLResponse(
        content,
        headers={
            "Cache-Control": "no-store",
            "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; frame-ancestors 'none'",
        },
    )


def _render_saved_process(config):
    from phase_agent.runtime.turn_process import render_process_panel

    with _message_lock:
        return render_process_panel(config)


async def process_panel(request):
    from starlette.concurrency import run_in_threadpool

    config = os.environ.get("PHASE_AGENT_RUNTIME_CONFIG")
    if not config:
        return HTMLResponse("当前没有项目运行。", status_code=503)
    content = await run_in_threadpool(_render_saved_process, config)
    return HTMLResponse(
        content,
        headers={
            "Cache-Control": "no-store",
            "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; frame-ancestors 'none'",
        },
    )


async def conversation_panel(request):
    return HTMLResponse(
        PAGE,
        headers={
            "Cache-Control": "no-store",
            "Content-Security-Policy": "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'",
        },
    )


app = Starlette(
    lifespan=lifespan,
    routes=[
        Route("/phase/conversations", conversation_panel, methods=["GET"]),
        Route("/phase/flow", flow_panel, methods=["GET"]),
        Route("/phase/process", process_panel, methods=["GET"]),
    ],
)

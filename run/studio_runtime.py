"""Agent Server owns one project handler shared by Studio and control pages."""
from contextlib import asynccontextmanager
import os
from pathlib import Path
import threading
from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.routing import Route

_handler = None
_message_lock = threading.RLock()


def send_project_message(text, thread_id):
    if _handler is None:
        raise RuntimeError("Studio project runtime has not started")
    from run.agent_api import _chat_content
    with _message_lock:
        return _chat_content(_handler([{"role": "user", "content": text}], conversation_id=thread_id))


@asynccontextmanager
async def lifespan(app):
    global _handler
    from run.agent_api import create_agent_runtime
    from run.agent_web_service import create_control_server
    config = os.environ.get("PHASE_AGENT_RUNTIME_CONFIG")
    if not config or not Path(config).is_file():
        raise RuntimeError("PHASE_AGENT_RUNTIME_CONFIG must identify an existing project configuration")
    if _handler is not None:
        raise RuntimeError("Studio runtime already owns a project handler")
    handler = create_agent_runtime(config)
    control = create_control_server(handler, config, port=int(os.environ.get("PHASE_CONTROL_PORT", "8765")))
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
    from run.chat_application import RunWorkflowChatHandler
    from run.flow_panel import render_flow_panel
    from execution_layer.step_runner.file_protocol import read_json
    handler = _handler
    seen = set()
    while not isinstance(handler, RunWorkflowChatHandler):
        if handler is None or id(handler) in seen:
            return HTMLResponse("配置尚未确认，请先在 Studio 完成配置。", status_code=503)
        seen.add(id(handler))
        handler = getattr(handler, "delegate", None)
    with handler.lock:
        state = read_json(handler.state_path, {}) or {}
        content = render_flow_panel(state)
    return HTMLResponse(content, headers={"Cache-Control": "no-store",
        "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; frame-ancestors 'none'"})


app = Starlette(lifespan=lifespan, routes=[Route("/phase/flow", flow_panel, methods=["GET"])])

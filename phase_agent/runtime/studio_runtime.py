"""Agent Server owns one project handler shared by Studio and control pages."""

from contextlib import asynccontextmanager
import asyncio
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

        with record_turn(os.environ["PHASE_AGENT_RUNTIME_CONFIG"], text, thread_id) as record:
            from phase_agent.runtime.loaded_code import (
                assert_loaded_code_current,
                LoadedCodeChangedError,
            )

            try:
                assert_loaded_code_current()
            except LoadedCodeChangedError as error:
                record["failure_summary"] = {
                    "type": type(error).__name__,
                    "stage": "运行版本核对",
                    "code": "manual_reload_required",
                    "message": str(error),
                    "modules": [
                        row
                        for row in error.report["modules"]
                        if row["status"] not in {"not_loaded", "matches"}
                    ],
                }
                process_event("本轮异常", record["failure_summary"])
                reply = (
                    "本轮未执行：运行中的代码与磁盘源码不一致。\n原因："
                    + str(error)
                    + "\n下一步：请在方便时手动重新启动项目服务加载一致版本；本次不会自动重启或重放请求。"
                )
                from phase_agent.runtime.interaction_reports import save_interaction

                try:
                    return save_interaction(
                        Path(os.environ["PHASE_AGENT_RUNTIME_CONFIG"]).resolve().parent,
                        text,
                        reply,
                        reply,
                        {},
                        record=record,
                    )
                except (OSError, ValueError, TypeError) as report_error:
                    process_event("交互记录保存失败", {"type": type(report_error).__name__})
                    return reply + "\n需注意：错误 Markdown 保存失败，请查看过程页诊断。"
            from phase_agent.runtime.studio_session_scope import studio_project_session

            with studio_project_session():
                reply = _chat_content(
                    _handler([{"role": "user", "content": text}], conversation_id=thread_id)
                )
            process_event("完整答复", {"reply": reply})
            full_reply = reply
            from phase_agent.runtime.response_preferences import detailed_response

            from phase_agent.runtime.turn_process import read_presentation_state
            from phase_agent.runtime.chat_state_presentation import format_epoch_reply
            from phase_agent.runtime.project_context import handler_project_context

            current_handler = project_handler(_handler)
            owner = current_handler or _handler
            owner_path = getattr(owner, "__dict__", {}).get("state_path")
            current_state = (
                read_presentation_state(owner_path) if isinstance(owner_path, (str, Path)) else {}
            )
            if not detailed_response():
                from phase_agent.runtime.studio_reply_presentation import format_turn_reply
                from phase_agent.runtime.configuration_chat import ConfigurationChatHandler

                if current_handler is not None:
                    reply = format_turn_reply(
                        reply,
                        current_state,
                        configuring=current_handler.config_delegate is not None,
                    )
                elif isinstance(_handler, ConfigurationChatHandler):
                    reply = format_turn_reply(reply, {}, configuring=True)
            # Use the same freshly read facts for both reply renderers.
            card = (
                handler_project_context(owner, current_state)
                if isinstance(owner_path, (str, Path))
                else None
            )
            configuring = current_handler is None or current_handler.config_delegate is not None
            reply = format_epoch_reply(
                reply, current_state, configuring=configuring, project_card=card
            )
            from phase_agent.runtime.runtime_diagnostics import diagnostic_notice

            reply += diagnostic_notice(os.environ["PHASE_AGENT_RUNTIME_CONFIG"])
            from phase_agent.runtime.interaction_reports import save_interaction

            try:
                reply = save_interaction(
                    Path(os.environ["PHASE_AGENT_RUNTIME_CONFIG"]).resolve().parent,
                    text,
                    full_reply,
                    reply,
                    current_state,
                    record=record,
                )
            except (OSError, ValueError, TypeError) as error:
                process_event(
                    "交互记录保存失败", {"type": type(error).__name__, "message": str(error)}
                )
                reply += "\n交互 Markdown 保存失败；保留本轮答复，未重放操作。"
            if current_handler is not None:
                current_handler.remember_turn_reply(record, reply)
            process_event("本轮答复", {"reply": reply})
            return reply


@asynccontextmanager
async def _project_lifespan(app):
    global _handler
    config = os.environ.get("PHASE_AGENT_RUNTIME_CONFIG")
    if not config or not await asyncio.to_thread(Path(config).is_file):
        raise RuntimeError(
            "PHASE_AGENT_RUNTIME_CONFIG must identify an existing project configuration"
        )
    if _handler is not None:
        raise RuntimeError("Studio runtime already owns a project handler")

    from phase_agent.runtime.studio_checkpoint_persistence import (
        install_checkpoint_persistence,
        flush_checkpoint_persistence,
    )

    from phase_agent.runtime.runtime_diagnostics import service_stage

    await asyncio.to_thread(service_stage, config, "加载 Studio 检查点")
    await asyncio.to_thread(install_checkpoint_persistence)

    def recovery_adapter():
        from phase_agent.runtime.studio_run_lifecycle import recover_studio_runs

        return recover_studio_runs

    recover_studio_runs = await asyncio.to_thread(recovery_adapter)
    await asyncio.to_thread(service_stage, config, "核对旧 Studio 会话恢复记录")
    await recover_studio_runs()

    from phase_agent.runtime.studio_payloads import install_studio_payload_views

    await asyncio.to_thread(install_studio_payload_views)

    def initialize():
        service_stage(config, "初始化项目处理器和控制页面")
        from phase_agent.runtime.agent_api import create_agent_runtime
        from phase_agent.runtime.agent_web_service import create_control_server

        handler = create_agent_runtime(config)
        control = create_control_server(
            handler, config, port=int(os.environ.get("PHASE_CONTROL_PORT", "8765"))
        )
        return handler, control

    handler, control = await asyncio.to_thread(initialize)
    worker = threading.Thread(target=control.serve_forever, daemon=True)
    _handler = handler
    worker.start()
    try:
        yield
    finally:
        _handler = None
        await asyncio.to_thread(flush_checkpoint_persistence)
        await asyncio.to_thread(control.shutdown)
        await asyncio.to_thread(control.server_close)
        await asyncio.to_thread(worker.join, timeout=5)


@asynccontextmanager
async def lifespan(app):
    from phase_agent.runtime.runtime_diagnostics import start_monitor, stop_monitor

    config = os.environ.get("PHASE_AGENT_RUNTIME_CONFIG")
    monitor = None
    if config and await asyncio.to_thread(Path(config).is_file):
        try:
            monitor = await asyncio.to_thread(start_monitor, config)
        except (OSError, RuntimeError):
            pass  # Observation is optional; original startup validation still applies.
    try:
        async with _project_lifespan(app):
            if monitor:
                await asyncio.to_thread(monitor.stage, "服务已就绪", ready=True)
            yield
    except Exception as error:
        if monitor:
            monitor.failure = {"stage": monitor.stage_name, "type": type(error).__name__}
            monitor.status = "failed"
        raise
    finally:
        if monitor:
            await asyncio.to_thread(stop_monitor, config, monitor)


def _render_flow_panel():
    from phase_agent.runtime.chat_application import RunWorkflowChatHandler
    from phase_agent.runtime.flow_panel import render_flow_panel
    from phase_agent.tools.step_runner.file_protocol import read_json

    handler = project_handler(_handler)
    if handler is None:
        return HTMLResponse("配置尚未确认，请先在 Studio 完成配置。", status_code=503)
    # Writers atomically replace state.json. Show the latest committed facts
    # without waiting for a long-running turn's business lock.
    state = read_json(handler.state_path, {}) or {}
    content = render_flow_panel(state)
    return HTMLResponse(
        content,
        headers={
            "Cache-Control": "no-store",
            "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; frame-ancestors 'none'",
        },
    )


async def flow_panel(request):
    from starlette.concurrency import run_in_threadpool

    return await run_in_threadpool(_render_flow_panel)


def _render_saved_process(config):
    from phase_agent.runtime.turn_process import render_process_panel

    # Diagnostics must remain available while the message worker holds its lock.
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

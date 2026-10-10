import asyncio
import importlib
import json
from pathlib import Path
from unittest.mock import Mock
import pytest
from phase_agent.runtime import studio_runtime


def test_direct_message_requires_started_runtime():
    with pytest.raises(RuntimeError, match="not started"):
        studio_runtime.send_project_message("继续", "t")


def test_lifespan_shares_handler_and_closes_control(tmp_path, monkeypatch):
    config = tmp_path / "agent_runtime.json"
    config.write_text("{}")
    monkeypatch.setenv("PHASE_AGENT_RUNTIME_CONFIG", str(config))
    handler = Mock(return_value="ok")
    control = Mock()
    monkeypatch.setattr("phase_agent.runtime.agent_api.create_agent_runtime", lambda path: handler)
    monkeypatch.setattr(
        "phase_agent.runtime.agent_web_service.create_control_server",
        lambda owner, path, port: control if owner is handler else None,
    )

    from unittest.mock import AsyncMock

    monkeypatch.setattr(
        "phase_agent.runtime.studio_run_lifecycle.recover_studio_runs", AsyncMock(return_value=0)
    )

    async def run():
        async with studio_runtime.lifespan(None):
            assert (
                studio_runtime.send_project_message("继续", "t")
                == "epoch0 · 配置修订中，等待确认\nok"
            )
            handler.assert_called_once_with(
                [{"role": "user", "content": "继续"}], conversation_id="t"
            )
        assert studio_runtime._handler is None

    asyncio.run(run())
    control.shutdown.assert_called_once()
    control.server_close.assert_called_once()


def test_configured_app_and_graph_share_the_same_runtime(tmp_path, monkeypatch):
    """Mirror Agent Server's normal-module import, not a direct lifespan call."""
    config = json.loads((Path(__file__).resolve().parents[1] / "langgraph.json").read_text())
    app_module, app_name = config["http"]["app"].rsplit(":", 1)
    assert (
        app_module == "phase_agent.runtime.studio_runtime"
    )  # File loading creates user_router_module.
    configured_app = getattr(importlib.import_module(app_module), app_name)
    graph_module, graph_name = config["graphs"]["phase_chat"].rsplit(":", 1)
    assert graph_module == "phase_agent.graphs.studio_chat_graph"
    graph = getattr(importlib.import_module(graph_module), graph_name)
    runtime_config = tmp_path / "agent_runtime.json"
    runtime_config.write_text("{}")
    monkeypatch.setenv("PHASE_AGENT_RUNTIME_CONFIG", str(runtime_config))
    monkeypatch.setenv("PHASE_STUDIO_RECEIPT_PATH", str(tmp_path / "gateway.json"))
    handler = Mock(return_value="synthetic status")
    control = Mock()
    monkeypatch.setattr("phase_agent.runtime.agent_api.create_agent_runtime", lambda path: handler)
    monkeypatch.setattr(
        "phase_agent.runtime.agent_web_service.create_control_server",
        lambda *args, **kwargs: control,
    )

    from unittest.mock import AsyncMock

    monkeypatch.setattr(
        "phase_agent.runtime.studio_run_lifecycle.recover_studio_runs", AsyncMock(return_value=0)
    )

    async def run():
        async with configured_app.router.lifespan_context(configured_app):
            result = graph.invoke(
                {"messages": [{"role": "user", "content": "现在是什么进度", "id": "m"}]},
                {"configurable": {"thread_id": "synthetic"}},
            )
            assert (
                result["messages"][-1].content == "epoch0 · 配置修订中，等待确认\nsynthetic status"
            )
        assert studio_runtime._handler is None

    asyncio.run(run())
    handler.assert_called_once()
    control.server_close.assert_called_once()


def test_process_page_reads_files_in_worker_thread(tmp_path, monkeypatch):
    import threading
    import phase_agent.runtime.turn_process

    monkeypatch.setenv("PHASE_AGENT_RUNTIME_CONFIG", str(tmp_path / "agent_runtime.json"))
    event_thread = threading.get_ident()
    readers = []

    def render(config):
        readers.append(threading.get_ident())
        return "saved process"

    monkeypatch.setattr(phase_agent.runtime.turn_process, "render_process_panel", render)
    response = asyncio.run(studio_runtime.process_panel(None))
    assert response.status_code == 200
    assert response.body == b"saved process"
    assert readers and readers[0] != event_thread


def test_conversation_management_page_exposes_pause_delete_and_same_origin_api():
    response = asyncio.run(studio_runtime.conversation_panel(None))
    page = response.body.decode("utf-8")
    assert response.status_code == 200
    assert "暂停" in page and "删除对话" in page
    assert "action=interrupt" in page and "'DELETE'" in page
    assert "if(runs.some(active))" in page
    assert "connect-src 'self'" in response.headers["content-security-policy"]

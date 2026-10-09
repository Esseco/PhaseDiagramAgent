import asyncio
import importlib
import json
from pathlib import Path
from unittest.mock import Mock
import pytest
from run import studio_runtime


def test_direct_message_requires_started_runtime():
    with pytest.raises(RuntimeError, match="not started"):
        studio_runtime.send_project_message("继续", "t")


def test_lifespan_shares_handler_and_closes_control(tmp_path, monkeypatch):
    config = tmp_path / "agent_runtime.json"
    config.write_text("{}")
    monkeypatch.setenv("PHASE_AGENT_RUNTIME_CONFIG", str(config))
    handler = Mock(return_value="ok")
    control = Mock()
    monkeypatch.setattr("run.agent_api.create_agent_runtime", lambda path: handler)
    monkeypatch.setattr("run.agent_web_service.create_control_server", lambda owner, path, port: control if owner is handler else None)

    async def run():
        async with studio_runtime.lifespan(None):
            assert studio_runtime.send_project_message("继续", "t") == "ok"
            handler.assert_called_once_with([{"role": "user", "content": "继续"}], conversation_id="t")
        assert studio_runtime._handler is None
    asyncio.run(run())
    control.shutdown.assert_called_once()
    control.server_close.assert_called_once()


def test_configured_app_and_graph_share_the_same_runtime(tmp_path, monkeypatch):
    """Mirror Agent Server's normal-module import, not a direct lifespan call."""
    config = json.loads((Path(__file__).resolve().parents[1] / "langgraph.json").read_text())
    app_module, app_name = config["http"]["app"].rsplit(":", 1)
    assert app_module == "run.studio_runtime"  # File loading creates user_router_module.
    configured_app = getattr(importlib.import_module(app_module), app_name)
    graph_module, graph_name = config["graphs"]["phase_chat"].rsplit(":", 1)
    assert graph_module == "orchestration.studio_chat_graph"
    graph = getattr(importlib.import_module(graph_module), graph_name)
    runtime_config = tmp_path / "agent_runtime.json"
    runtime_config.write_text("{}")
    monkeypatch.setenv("PHASE_AGENT_RUNTIME_CONFIG", str(runtime_config))
    monkeypatch.setenv("PHASE_STUDIO_RECEIPT_PATH", str(tmp_path / "gateway.json"))
    handler = Mock(return_value="synthetic status")
    control = Mock()
    monkeypatch.setattr("run.agent_api.create_agent_runtime", lambda path: handler)
    monkeypatch.setattr("run.agent_web_service.create_control_server", lambda *args, **kwargs: control)

    async def run():
        async with configured_app.router.lifespan_context(configured_app):
            result = graph.invoke({"messages": [{"role": "user", "content": "现在是什么进度", "id": "m"}]},
                                  {"configurable": {"thread_id": "synthetic"}})
            assert result["messages"][-1].content == "synthetic status"
        assert studio_runtime._handler is None

    asyncio.run(run())
    handler.assert_called_once()
    control.server_close.assert_called_once()

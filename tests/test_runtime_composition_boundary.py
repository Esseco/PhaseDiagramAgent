import ast
from pathlib import Path
from unittest.mock import patch

from phase_agent.runtime import agent_api as open_webui_api, runtime_composition


def test_runtime_composition_does_not_import_chat_entry():
    tree = ast.parse(Path(runtime_composition.__file__).read_text(encoding="utf-8"))
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert "phase_agent.runtime.agent_api" not in imports
    assert "http.server" not in imports


def test_old_entry_passes_explicit_factories_and_restart_callback():
    sentinel = object()
    with patch("phase_agent.runtime.runtime_composition.compose_runtime", return_value=sentinel) as compose:
        assert open_webui_api.create_agent_runtime("runtime.json") is sentinel
    args, kwargs = compose.call_args
    assert args == ("runtime.json",)
    assert kwargs["chat_handler_factory"] is open_webui_api.RunWorkflowChatHandler
    assert kwargs["model_switcher_factory"] is open_webui_api._make_deepseek_model_switcher
    assert kwargs["runtime_factory"] is open_webui_api.create_agent_runtime


def test_codex_default_factory_requests_no_project_model_capability():
    sentinel = object()
    with patch("phase_agent.runtime.runtime_composition.compose_runtime", return_value=sentinel) as compose:
        assert open_webui_api.create_agent_runtime("runtime.json",operator="codex") is sentinel
    assert compose.call_args.kwargs["operator"]=="codex"


def test_offline_default_factory_constructs_codex_runtime(monkeypatch):
    from types import SimpleNamespace
    from threading import RLock
    from phase_agent.runtime.codex import offline
    calls=[]
    def factory(path, *, operator):
        assert operator=="codex"
        calls.append(path)
        return SimpleNamespace(state_path="synthetic/state.json",lock=RLock(),workflow_kwargs={})
    monkeypatch.setattr(offline,"create_agent_runtime",factory)
    agent=offline.CodexOffline("synthetic.json")
    assert calls==["synthetic.json"]
    from phase_agent.runtime.codex.no_project_model import NoProjectModel
    assert isinstance(agent.handler.workflow_kwargs["agent_client"],NoProjectModel)


def test_codex_composition_does_not_construct_client_or_switcher(tmp_path):
    from tests.test_agent_api import OpenWebUIAPITest
    runtime=OpenWebUIAPITest()._runtime_files(tmp_path)
    with patch("phase_agent.runtime.runtime_clients.create_optional_runtime_client", side_effect=AssertionError("project client constructed")), patch("phase_agent.runtime.agent_api._make_deepseek_model_switcher", side_effect=AssertionError("switcher constructed")):
        handler=open_webui_api.create_agent_runtime(runtime,operator="codex")
    from phase_agent.runtime.codex.no_project_model import NoProjectModel
    assert isinstance(handler.workflow_kwargs["agent_client"],NoProjectModel)

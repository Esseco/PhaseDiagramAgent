import ast
from pathlib import Path
from unittest.mock import patch

from run import agent_api as open_webui_api, runtime_composition


def test_runtime_composition_does_not_import_chat_entry():
    tree = ast.parse(Path(runtime_composition.__file__).read_text(encoding="utf-8"))
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert "run.agent_api" not in imports
    assert "http.server" not in imports


def test_old_entry_passes_explicit_factories_and_restart_callback():
    sentinel = object()
    with patch("run.runtime_composition.compose_runtime", return_value=sentinel) as compose:
        assert open_webui_api.create_agent_runtime("runtime.json") is sentinel
    args, kwargs = compose.call_args
    assert args == ("runtime.json",)
    assert kwargs["chat_handler_factory"] is open_webui_api.RunWorkflowChatHandler
    assert kwargs["model_switcher_factory"] is open_webui_api._make_deepseek_model_switcher
    assert kwargs["runtime_factory"] is open_webui_api.create_agent_runtime

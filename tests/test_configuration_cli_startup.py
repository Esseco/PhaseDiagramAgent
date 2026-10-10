"""A new project's configuration handler must be accepted by the HTTP adapter."""
from types import SimpleNamespace
from unittest.mock import Mock
from phase_agent.runtime import agent_api
from phase_agent.runtime.chat_application import RunWorkflowChatHandler


def test_cli_accepts_configuration_before_search(tmp_path, monkeypatch):
    workflow = RunWorkflowChatHandler({"state_path": tmp_path / "state.json"})
    configuration = SimpleNamespace(runtime_factory=lambda: workflow)
    monkeypatch.setenv("OPENWEBUI_TOOL_TOKEN", "tool-token-1234567890")
    monkeypatch.setenv("OPENWEBUI_CONTROL_TOKEN", "control-token-1234567890")
    monkeypatch.setattr(agent_api, "_load_factory", lambda reference: configuration)
    server = Mock()
    create = Mock(return_value=server)
    monkeypatch.setattr(agent_api, "create_server", create)
    agent_api.main(["--handler-factory", "fixture:create"])
    assert create.call_args.args[0] is configuration
    server.serve_forever.assert_called_once()
    assert configuration.runtime_factory() is workflow
    assert workflow.decision_backend == "langgraph"

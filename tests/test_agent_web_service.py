from types import SimpleNamespace
from unittest.mock import Mock, patch
import pytest
from run.agent_web_service import configure_decision_backend, serve_unified, DeferredControl


def test_backend_survives_configuration_transition(tmp_path):
    from run.agent_api import RunWorkflowChatHandler
    workflow = RunWorkflowChatHandler({"state_path": tmp_path / "state.json"})
    configuration = SimpleNamespace(runtime_factory=lambda: workflow)
    configure_decision_backend(configuration, "langgraph")
    assert configuration.runtime_factory() is workflow
    assert workflow.decision_backend == "langgraph"


def test_service_closes_control_listener_on_ui_failure():
    control, ui = Mock(), Mock()
    ui.run.side_effect = RuntimeError("UI bind failed")
    with patch("uvicorn.Server", return_value=ui):
        with pytest.raises(RuntimeError, match="UI bind"):
            serve_unified(object(), control, port=7932)
    control.shutdown.assert_called_once()
    control.server_close.assert_called_once()


def test_control_follows_confirmed_delegate_without_new_runtime(tmp_path):
    from run.agent_api import RunWorkflowChatHandler
    configuration = SimpleNamespace(delegate=None)
    control = DeferredControl(configuration)
    with pytest.raises(ValueError, match="配置尚未确认"):
        control.status()
    workflow = RunWorkflowChatHandler({"state_path": tmp_path / "state.json"})
    configuration.delegate = workflow
    with patch("run.local_agent_control.LocalAgentControl") as factory:
        factory.return_value.status.return_value = {"status": "ok"}
        assert control.status() == {"status": "ok"}
    factory.assert_called_once_with(workflow)

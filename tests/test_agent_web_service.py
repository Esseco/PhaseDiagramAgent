from types import SimpleNamespace
from unittest.mock import patch
import pytest
from phase_agent.runtime.agent_web_service import configure_decision_backend, DeferredControl


def test_backend_survives_configuration_transition(tmp_path):
    from phase_agent.runtime.agent_api import RunWorkflowChatHandler
    workflow = RunWorkflowChatHandler({"state_path": tmp_path / "state.json"})
    configuration = SimpleNamespace(runtime_factory=lambda: workflow)
    configure_decision_backend(configuration, "langgraph")
    assert configuration.runtime_factory() is workflow
    assert workflow.decision_backend == "langgraph"



def test_control_follows_confirmed_delegate_without_new_runtime(tmp_path):
    from phase_agent.runtime.agent_api import RunWorkflowChatHandler
    configuration = SimpleNamespace(delegate=None)
    control = DeferredControl(configuration)
    with pytest.raises(ValueError, match="配置尚未确认"):
        control.status()
    workflow = RunWorkflowChatHandler({"state_path": tmp_path / "state.json"})
    configuration.delegate = workflow
    with patch("phase_agent.runtime.local_agent_control.LocalAgentControl") as factory:
        factory.return_value.status.return_value = {"status": "ok"}
        assert control.status() == {"status": "ok"}
    factory.assert_called_once_with(workflow)

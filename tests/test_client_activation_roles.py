from unittest.mock import patch

from execution_layer.step_runner.file_protocol import write_json
from run.agent_api import RunWorkflowChatHandler, _make_deepseek_key_setup


def test_key_activation_preserves_search_and_intent_roles(tmp_path):
    runtime = tmp_path / "runtime.json"
    write_json(runtime, {"deepseek": {}})
    handler = RunWorkflowChatHandler({"state_path": str(tmp_path / "state.json")})
    clients = [object(), object()]
    with patch("run.deepseek_setup.test_and_save_api_key",
               side_effect=lambda key, **kwargs: kwargs["activate_client"](key)), \
         patch("decision_layer.agent.create_deepseek_client.create_deepseek_client",
               side_effect=clients) as factory:
        _make_deepseek_key_setup(handler, runtime)("mock-key")
    search, intent = [call.kwargs for call in factory.call_args_list]
    assert search["model"] == "deepseek-v4-pro"
    assert search["routine_max_tokens"] == 1600
    assert intent["max_tokens"] == 120
    assert intent["thinking"] == "disabled"
    assert handler.workflow_kwargs["agent_client"] is clients[0]
    assert handler.config_intent_client is clients[1]


def test_existing_setup_callback_reads_current_runtime_descriptor(tmp_path):
    old, current = tmp_path / "old.json", tmp_path / "current.json"
    write_json(old, {"deepseek": {"model": "old-model"}})
    write_json(current, {"deepseek": {"model": "current-model"}})
    handler = RunWorkflowChatHandler({"state_path": str(tmp_path / "state.json")})
    handler.runtime_config_path = current
    with patch("run.deepseek_setup.test_and_save_api_key", return_value={"status": "mock"}) as save:
        setup = _make_deepseek_key_setup(handler, old)
        setup("mock-key")
    assert save.call_args.kwargs["settings"]["model"] == "current-model"


def test_key_activation_targets_current_configuration_delegate(tmp_path):
    from run.configuration_chat import ConfigurationChatHandler
    runtime = tmp_path / "runtime.json"
    write_json(runtime, {"deepseek": {"model": "delegate-model"}})
    handler = RunWorkflowChatHandler({"state_path": str(tmp_path / "state.json")})
    delegate = object.__new__(ConfigurationChatHandler)
    delegate.runtime_config_path = runtime
    delegate.agent_client = None
    handler.config_delegate = delegate
    client = object()
    with patch("run.deepseek_setup.test_and_save_api_key",
               side_effect=lambda key, **kwargs: kwargs["activate_client"](key)), \
         patch("decision_layer.agent.create_deepseek_client.create_deepseek_client",
               return_value=client) as factory:
        _make_deepseek_key_setup(handler, tmp_path / "missing-old.json")("mock-key")
    assert delegate.agent_client is client
    assert "agent_client" not in handler.workflow_kwargs
    assert factory.call_args.kwargs["model"] == "delegate-model"

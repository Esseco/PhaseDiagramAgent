from unittest.mock import Mock, patch

from run.local_project_launcher import start_agent


def test_launcher_starts_langgraph_ui_and_checks_both_listeners(tmp_path):
    runtime = tmp_path / "runtime.json"
    process = Mock()
    process.poll.return_value = None
    response = Mock()
    response.status = 200
    context = Mock()
    context.__enter__ = Mock(return_value=response)
    context.__exit__ = Mock(return_value=False)
    with patch("run.local_project_launcher._load_config", return_value={}), \
         patch("run.local_project_launcher._port_is_free", return_value=True), \
         patch("run.local_project_launcher.local_service_tokens", return_value=("a" * 24, "b" * 24)), \
         patch("run.local_project_launcher.subprocess.Popen", return_value=process) as popen, \
         patch("run.local_project_launcher.urlopen", return_value=context) as get:
        result = start_agent(runtime)
    assert result == (process, "a" * 24, False)
    command = popen.call_args.args[0]
    assert "run.studio_service" in command
    assert "run.pydantic_web_chat" not in command
    assert "run.agent_api" not in command
    assert "--control-port" in command
    assert popen.call_args.kwargs["env"]["PHASE_WEB_PASSWORD"] == "a" * 24
    assert get.call_count == 2


def test_launcher_refuses_occupied_ui_port(tmp_path):
    import pytest
    with patch("run.local_project_launcher._load_config", return_value={}), \
         patch("run.local_project_launcher._port_is_free", side_effect=[True, False]), \
         patch("run.local_project_launcher.local_service_tokens", return_value=("a" * 24, "b" * 24)), \
         patch("run.local_project_launcher.subprocess.Popen") as popen:
        with pytest.raises(RuntimeError, match="聊天端口"):
            start_agent(tmp_path / "runtime.json")
    popen.assert_not_called()

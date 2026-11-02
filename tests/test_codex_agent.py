from pathlib import Path
from unittest.mock import Mock

from phase_agent.runtime.codex.agent import CodexAgent, build_service_command


def test_build_service_command_targets_one_runtime_config():
    command = build_service_command(
        Path(r"E:\0-PhaseDiagram\O3-1ele-NaCrO2\agent_runtime.json"),
        python_executable=r"C:\py1\python.exe",
        control_port=8765,
        studio_port=2024,
    )
    assert command == [
        r"C:\py1\python.exe",
        "-m",
        "phase_agent.runtime.studio_service",
        "--runtime-config",
        r"E:\0-PhaseDiagram\O3-1ele-NaCrO2\agent_runtime.json",
        "--control-port",
        "8765",
        "--port",
        "2024",
    ]


def test_start_reuses_existing_project_service_without_spawning():
    identity = {
        "status": "ready",
        "runtime_config": r"E:\0-PhaseDiagram\O3-1ele-NaCrO2\agent_runtime.json",
        "control_port": 8765,
        "studio_port": 2024,
    }
    popen = Mock()
    agent = CodexAgent(
        identity["runtime_config"],
        process_factory=popen,
        service_reader=Mock(return_value=identity),
    )

    result = agent.start()

    assert result == {"status": "already_running", "service": identity}
    popen.assert_not_called()


def test_start_spawns_when_project_service_is_absent():
    popen = Mock(return_value=Mock())
    agent = CodexAgent(
        r"E:\0-PhaseDiagram\O3-1ele-NaCrO2\agent_runtime.json",
        process_factory=popen,
        service_reader=Mock(
            side_effect=[
                None,
                {
                    "status": "ready",
                    "control_port": 8765,
                    "studio_port": 2024,
                },
            ]
        ),
        readiness_wait=0,
    )

    result = agent.start()

    assert result["status"] == "started"
    assert result["service"]["status"] == "ready"
    popen.assert_called_once()
    assert popen.call_args.args[0][2] == "phase_agent.runtime.studio_service"


def test_stop_requires_exact_project_identity():
    reader = Mock(return_value=None)
    agent = CodexAgent(
        r"E:\0-PhaseDiagram\O3-1ele-NaCrO2\agent_runtime.json",
        service_reader=reader,
    )

    assert agent.stop() == {"status": "not_running"}

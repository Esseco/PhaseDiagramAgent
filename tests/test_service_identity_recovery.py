from types import SimpleNamespace
from phase_agent.runtime.project_service import read_service


def test_stopped_record_recovers_only_exact_project_listener(tmp_path, monkeypatch):
    import psutil

    monkeypatch.setattr("phase_agent.runtime.project_service.service_healthy", lambda value: True)
    config = tmp_path / "agent_runtime.json"
    directory = tmp_path / "workflow_state" / "studio"
    directory.mkdir(parents=True)
    (directory / "service.json").write_text('{"status":"stopped"}')
    command = [
        "python",
        "-m",
        "phase_agent.runtime.studio_service",
        "--runtime-config",
        str(config),
        "--port",
        "2025",
        "--control-port",
        "8766",
    ]
    process = SimpleNamespace(
        pid=12,
        info={"cmdline": command},
        children=lambda recursive: [SimpleNamespace(pid=13)],
        create_time=lambda: 100,
    )
    monkeypatch.setattr(psutil, "process_iter", lambda attrs: [process])
    ports = [2025, 8766]
    monkeypatch.setattr(
        psutil,
        "net_connections",
        lambda kind: [
            SimpleNamespace(pid=13, status="LISTEN", laddr=SimpleNamespace(ip="127.0.0.1", port=p))
            for p in ports
        ],
    )
    assert read_service(config)["pid"] == 12
    ports.pop()
    assert read_service(config) is None
    ports.append(8766)
    command[command.index("--runtime-config") + 1] = str(tmp_path / "other.json")
    assert read_service(config) is None

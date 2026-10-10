"""Project-owned service identity, ports and process-lifetime lock."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket


def service_directory(config_path):
    return Path(config_path).resolve().parent / "workflow_state" / "studio"


def available_port(preferred, excluded=()):
    for port in range(preferred, min(preferred + 200, 65536)):
        if port in excluded:
            continue
        with socket.socket() as listener:
            try:
                listener.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("没有可用的项目服务端口")


@contextmanager
def project_lock(config_path, *, lock_name="service.lock"):
    """OS lock survives stale files and is released even after a process crash."""
    directory = service_directory(config_path)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / lock_name).open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError("该项目已经运行，请打开已有 Studio，不要重复启动") from None
        try:
            yield directory
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def write_service(config_path, value):
    directory = service_directory(config_path)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "service.json"
    temporary = directory / f"service-{os.getpid()}.tmp"
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)


def service_healthy(value):
    """Both loopback endpoints must answer; HTTP proxies never participate."""
    from urllib.request import ProxyHandler, build_opener

    try:
        opener = build_opener(ProxyHandler({}))
        for port, suffix in ((value["studio_port"], "info"), (value["control_port"], "health")):
            with opener.open(f"http://127.0.0.1:{int(port)}/{suffix}", timeout=0.5) as response:
                if response.status != 200:
                    return False
        return True
    except (OSError, ValueError, KeyError):
        return False


def service_ports_owned(process, value):
    """Healthy endpoints must also belong to this exact service process tree."""
    import psutil

    try:
        command = process.cmdline()
        expected = {int(value["studio_port"]), int(value["control_port"])}
        if (
            int(command[command.index("--port") + 1]) != value["studio_port"]
            or int(command[command.index("--control-port") + 1]) != value["control_port"]
        ):
            return False
        owners = {process.pid, *(child.pid for child in process.children(recursive=True))}
        listening = {
            row.laddr.port
            for row in psutil.net_connections(kind="tcp")
            if row.pid in owners and row.status == "LISTEN" and row.laddr.ip in {"127.0.0.1", "::1"}
        }
        return expected.issubset(listening)
    except (OSError, ValueError, KeyError, IndexError, psutil.Error):
        return False


def read_service(config_path, *, require_ready=True):
    import psutil

    try:
        value = json.loads(
            (service_directory(config_path) / "service.json").read_text(encoding="utf-8")
        )
        process = psutil.Process(value["pid"])
        command = process.cmdline()
        valid = (
            abs(process.create_time() - value["process_started"]) <= 0.1
            and Path(value["runtime_config"]).resolve() == Path(config_path).resolve()
            and "phase_agent.runtime.studio_service" in command
            and "--runtime-config" in command
            and Path(command[command.index("--runtime-config") + 1]).resolve()
            == Path(config_path).resolve()
        )
        if valid and (
            not require_ready
            or (
                value.get("status") == "ready"
                and service_ports_owned(process, value)
                and service_healthy(value)
            )
        ):
            return value
    except (OSError, ValueError, KeyError, IndexError, psutil.Error):
        pass
    return recover_service_identity(config_path, require_ready=require_ready)


def prepare_studio_config(config_path):
    directory = service_directory(config_path)
    directory.mkdir(parents=True, exist_ok=True)
    code = Path(__file__).resolve().parents[2]
    target = directory / "langgraph.json"
    value = {
        "dependencies": [str(code)],
        "graphs": {
            "phase_chat": "phase_agent.graphs.studio_chat_graph:graph",
            "project_status": "phase_agent.graphs.studio_status_graph:graph",
        },
        "http": {"app": "phase_agent.runtime.studio_runtime:app"},
    }
    target.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return target


def recover_service_identity(config_path, *, require_ready=True):
    """Recover a lost record only from an exact project owner and its listening children."""
    import psutil

    matches = []
    selected = Path(config_path).resolve()
    for process in psutil.process_iter(["cmdline"]):
        try:
            command = process.info["cmdline"] or []
            if "phase_agent.runtime.studio_service" not in command:
                continue
            configured = Path(command[command.index("--runtime-config") + 1]).resolve()
            if configured != selected:
                continue
            studio_port = int(command[command.index("--port") + 1])
            control_port = int(command[command.index("--control-port") + 1])
            owners = {process.pid, *(child.pid for child in process.children(recursive=True))}
            listening = {
                connection.laddr.port
                for connection in psutil.net_connections(kind="tcp")
                if connection.pid in owners
                and connection.status == "LISTEN"
                and connection.laddr.ip in {"127.0.0.1", "::1"}
            }
            if require_ready and not {studio_port, control_port}.issubset(listening):
                continue
            matches.append(
                {
                    "pid": process.pid,
                    "process_started": process.create_time(),
                    "runtime_config": str(selected),
                    "project_name": selected.parent.name,
                    "control_port": control_port,
                    "studio_port": studio_port,
                    "status": "ready",
                }
            )
        except (OSError, ValueError, IndexError, psutil.Error):
            continue
    if len(matches) != 1:
        return None
    value = matches[0]
    return value if not require_ready or service_healthy(value) else None

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
def project_lock(config_path):
    """OS lock survives stale files and is released even after a process crash."""
    directory = service_directory(config_path)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "service.lock").open("a+b") as stream:
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


def read_service(config_path):
    import psutil
    try:
        value = json.loads((service_directory(config_path) / "service.json").read_text(encoding="utf-8"))
        process = psutil.Process(value["pid"])
        if abs(process.create_time() - value["process_started"]) > .1:
            return None
        if Path(value["runtime_config"]).resolve() != Path(config_path).resolve():
            return None
        return value if value.get("status") == "ready" else None
    except (OSError, ValueError, KeyError, psutil.Error):
        return None


def prepare_studio_config(config_path):
    directory = service_directory(config_path)
    directory.mkdir(parents=True, exist_ok=True)
    code = Path(__file__).resolve().parents[1]
    target = directory / "langgraph.json"
    value = {"dependencies": [str(code)],
             "graphs": {"phase_chat": "orchestration.studio_chat_graph:graph"},
             "http": {"app": "run.studio_runtime:app"}}
    target.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return target

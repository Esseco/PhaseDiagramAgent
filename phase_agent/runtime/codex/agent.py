"""Project-scoped lifecycle control for the existing local Agent service."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable

from phase_agent.runtime.project_service import read_service, service_directory


class CodexAgentError(RuntimeError):
    """A lifecycle request cannot be safely completed."""


def build_service_command(
    runtime_config: Path,
    *,
    python_executable: str,
    control_port: int,
    studio_port: int,
) -> list[str]:
    return [
        python_executable,
        "-m",
        "phase_agent.runtime.studio_service",
        "--runtime-config",
        str(Path(runtime_config).resolve()),
        "--control-port",
        str(control_port),
        "--port",
        str(studio_port),
    ]


class CodexAgent:
    """Start, stop, and address one project-owned Agent service."""

    def __init__(
        self,
        runtime_config: str | Path,
        *,
        python_executable: str | None = None,
        control_port: int = 8765,
        studio_port: int = 2024,
        startup_deadline: float = 60.0,
        readiness_wait: float = 0.5,
        process_factory: Callable[..., Any] = subprocess.Popen,
        service_reader: Callable[..., dict | None] = read_service,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        self.runtime_config = Path(runtime_config).resolve()
        self.python_executable = python_executable or sys.executable
        self.control_port = int(control_port)
        self.studio_port = int(studio_port)
        self.startup_deadline = float(startup_deadline)
        self.readiness_wait = float(readiness_wait)
        self.process_factory = process_factory
        self.service_reader = service_reader
        self._default_service_reader = service_reader is read_service
        self.sleeper = sleeper

    def _service(self, *, require_ready: bool) -> dict | None:
        service = self.service_reader(self.runtime_config, require_ready=require_ready)
        if service or not self._default_service_reader:
            return service
        # Preserve a project-owned identity when the launcher parent exited but
        # its Studio child is still serving; _owned_process validates the child.
        import json

        identity = service_directory(self.runtime_config) / "service.json"
        try:
            value = json.loads(identity.read_text(encoding="utf-8"))
            if (
                value.get("runtime_config")
                and Path(value["runtime_config"]).resolve() == self.runtime_config
            ):
                import psutil

                child = psutil.Process(int(value.get("child_pid")))
                if "phase_agent.runtime.studio_server" in child.cmdline():
                    return value
        except (OSError, ValueError, TypeError, psutil.Error):
            pass
        return None

    def start(self) -> dict[str, Any]:
        existing = self._service(require_ready=True)
        if existing:
            return {"status": "already_running", "service": existing}
        if not self.runtime_config.is_file():
            raise CodexAgentError(f"runtime 配置不存在：{self.runtime_config}")
        command = build_service_command(
            self.runtime_config,
            python_executable=self.python_executable,
            control_port=self.control_port,
            studio_port=self.studio_port,
        )
        flags = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0)
            | getattr(subprocess, "DETACHED_PROCESS", 0)
        )
        process = self.process_factory(
            command,
            cwd=str(Path(__file__).resolve().parents[3]),
            creationflags=flags,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
        deadline = time.monotonic() + self.startup_deadline
        while time.monotonic() <= deadline:
            service = self._service(require_ready=True)
            if service:
                return {"status": "started", "service": service}
            if process.poll() is not None:
                raise CodexAgentError(f"Agent 服务启动失败，退出码：{process.poll()}")
            if self.readiness_wait:
                self.sleeper(self.readiness_wait)
        self._terminate_spawned_process(process)
        raise CodexAgentError("Agent 服务启动超时；请检查项目 service 日志")

    def stop(self) -> dict[str, Any]:
        service = self._service(require_ready=False)
        if not service:
            return {"status": "not_running"}
        process = self._owned_process(service)
        from phase_agent.runtime.studio_run_lifecycle import interrupt_owned_studio

        interrupt_owned_studio(process)
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        return {"status": "stopped", "pid": process.pid}

    def takeover(self) -> dict[str, Any]:
        stopped = self.stop()
        started = self.start()
        return {"status": "restarted", "stopped": stopped, "started": started}

    def _owned_process(self, service: dict[str, Any]):
        import psutil

        expected = str(self.runtime_config)

        # On Windows the detached Studio child is the process that owns the
        # listening service.  The launcher parent may have exited and its PID
        # can later be reused by an unrelated process, so validate the child
        # first and never trust the recorded parent PID by itself.
        try:
            child = psutil.Process(int(service["child_pid"]))
            command = child.cmdline()
            config = Path(command[command.index("--config") + 1]).resolve()
            owned_config = (service_directory(self.runtime_config) / "langgraph.json").resolve()
            if (
                "phase_agent.runtime.studio_server" in command
                and config == owned_config
                and int(command[command.index("--port") + 1]) == self.studio_port
                and int(service.get("control_port")) == self.control_port
                and int(service.get("studio_port")) == self.studio_port
            ):
                return child
        except (KeyError, ValueError, OSError, IndexError, psutil.Error):
            pass

        # A still-running launcher is accepted only when its creation time and
        # full command line both match the durable identity.
        try:
            process = psutil.Process(int(service["pid"]))
            command = process.cmdline()
            if (
                abs(process.create_time() - float(service["process_started"])) <= 0.1
                and "phase_agent.runtime.studio_service" in command
                and "--runtime-config" in command
                and str(Path(command[command.index("--runtime-config") + 1]).resolve()) == expected
            ):
                if (
                    int(service.get("control_port")) != self.control_port
                    or int(service.get("studio_port")) != self.studio_port
                ):
                    raise CodexAgentError("服务端口与指定项目不匹配，拒绝停止")
                return process
        except (KeyError, ValueError, OSError, IndexError, psutil.Error):
            pass
        raise CodexAgentError("找不到与项目匹配的 Agent 服务进程")

    @staticmethod
    def _terminate_spawned_process(process):
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="管理一个指定项目的本地 Agent 服务")
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--control-port", type=int, default=8765)
    parser.add_argument("--studio-port", type=int, default=2024)
    parser.add_argument("--python", dest="python_executable")
    parser.add_argument("command", choices=("start", "stop", "takeover"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        agent = CodexAgent(
            args.runtime_config,
            python_executable=args.python_executable,
            control_port=args.control_port,
            studio_port=args.studio_port,
        )
        result = getattr(agent, args.command)()
    except CodexAgentError as error:
        print(f'{{"status": "error", "message": {error!r}}}', file=sys.stderr)
        return 1
    import json

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

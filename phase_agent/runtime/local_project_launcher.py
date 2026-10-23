"""Project picker and one-click launcher for the LangGraph Agent."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import time
from urllib.request import Request, urlopen
import webbrowser

from phase_agent.runtime.local_service_tokens import local_service_tokens
from phase_agent.runtime.project_registry import read_project_registry as read_project_registry
from phase_agent.runtime.project_registry import edit_project_registry


from phase_agent.configuration.runtime.project_template import (
    PROJECT_CONFIG_NAME,
    DEFAULT_PROJECT_SETTINGS,
)


def registry_path() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    if not base:
        raise RuntimeError("找不到本机 LOCALAPPDATA；请指定项目配置文件手动启动")
    return Path(base) / "PhaseSearchAgent" / "projects.json"


def register_project(path: Path, *, registry: Path | None = None) -> list[str]:
    """Register a config without moving any state or overwriting existing projects."""
    registry = registry or registry_path()
    target = path.resolve()
    if not target.is_file():
        raise FileNotFoundError(f"运行时配置不存在：{target}")
    new_state, new_ledger = _data_paths(target)
    with edit_project_registry(registry) as projects:
        for old in projects:
            old_path = Path(old)
            if old_path.resolve() == target:
                continue
            if target.parent == old_path.resolve().parent:
                raise ValueError("每个文件夹只能注册一个项目配置")
            if not old_path.is_file():
                continue
            try:
                old_state, old_ledger = _data_paths(old_path)
            except (OSError, ValueError) as error:
                import logging

                logging.getLogger(__name__).warning(
                    "旧项目配置暂不可读，保留原文件和清单条目，不阻止其他独立项目：%s；%s",
                    old_path,
                    error,
                )
                continue
            if old_state == new_state or old_ledger == new_ledger:
                raise ValueError("两个项目指向同一 state 或台账；请为新项目选择独立工作区")
        validate_project_paths(target)
        if str(target) not in projects:
            projects.append(str(target))
    return projects


def validate_project_paths(path):
    """Writable runtime and confirmed science storage must stay in this folder."""
    path = Path(path).resolve()
    root = path.parent
    settings = _load_config(path)
    system_root = root if (root / "system.json").is_file() else root.parent.parent
    descriptor = _load_optional_json(system_root / "system.json")
    diagram = (settings.get("shared_dft") or {}).get("diagram_id")
    registered = bool(
        diagram
        and diagram in descriptor.get("diagrams", [])
        and path
        in {
            system_root / PROJECT_CONFIG_NAME,
            system_root / "diagrams" / diagram / PROJECT_CONFIG_NAME,
        }
    )
    canonical = system_root / "diagrams" / diagram if registered else root
    # Default-diagram aliases may resolve to old physical directories on Windows.
    owned = [canonical.resolve()]
    if registered:
        owned.extend(
            (canonical / name).resolve()
            for name in (
                "workflow_state",
                "parameters",
                "structures",
                "submissions",
                "analysis_outputs",
                "open_webui_runs",
            )
        )
    if registered and diagram == descriptor.get("default_diagram"):
        # Existing default diagrams retain their physical storage at system level.
        # Other diagrams must never gain access to these legacy directories.
        owned.extend(
            (system_root / name).resolve()
            for name in (
                "workflow_state",
                "parameters",
                "structures",
                "submissions",
                "analysis_outputs",
                "open_webui_runs",
            )
        )
    keys = (
        "state_path",
        "ledger_path",
        "config_session_path",
        "editable_config_draft_path",
        "approval_directory",
        "local_action_directory",
        "new_runs_directory",
        "phase_diagram_directory",
        "branch_energy_pool_ledger_path",
    )
    for key in keys:
        if settings.get(key):
            target = _resolve(root, settings[key])
            if not any(target.is_relative_to(owner) for owner in owned):
                raise ValueError(f"项目写入路径 {key} 超出项目文件夹：{target}")
    session = _load_optional_json(
        _resolve(root, settings.get("config_session_path", "parameters/config_session.json"))
    )
    for config in (
        session.get("config") or {},
        (session.get("confirmed_snapshot") or {}).get("config") or {},
        {"storage": settings.get("runtime_storage_override") or {}},
    ):
        if (
            registered
            and diagram == descriptor.get("default_diagram")
            and settings.get("runtime_storage_override")
        ):
            config = {"storage": settings["runtime_storage_override"]}
        if settings.get("local_path_relocations") and not registered:
            from phase_agent.configuration.session.relocate_workspace_paths import (
                relocate_workspace_paths,
            )

            config = relocate_workspace_paths(config, settings["local_path_relocations"], root)
        storage = config.get("storage") or {}
        if storage.get("workspace_root"):
            owner = _resolve(root, storage["workspace_root"])
            if owner != canonical.resolve() and not (registered and owner == system_root):
                raise ValueError("配置工作区必须与项目文件夹一致；请先核对旧项目路径")


def create_project(
    workspace_root: Path, *, registry: Path | None = None, execution_mode="debug"
) -> Path:
    """Create an isolated runtime and editable science draft after GUI path selection."""
    if execution_mode not in {"debug", "automatic"}:
        raise ValueError("未知运行模式")
    from phase_agent.configuration.session.validate_workspace_root import validate_workspace_root

    validate_workspace_root(str(workspace_root))
    root = workspace_root.expanduser().resolve()
    # A new project may contain reference structures, but never prior runtime facts.
    existing = [
        root / name
        for name in (
            "workflow_state",
            "agent_memory",
            "submissions",
            "analysis_outputs",
            "state.json",
            "phase_data.json",
        )
        if (root / name).exists() and (not (root / name).is_dir() or any((root / name).iterdir()))
    ]
    if existing:
        names = "、".join(path.name for path in existing)
        raise FileExistsError(
            f"该目录已有项目记录（{names}）；请使用“添加已有项目”或选择新的工作区"
        )
    root.mkdir(parents=True, exist_ok=True)
    target = root / PROJECT_CONFIG_NAME
    session_path = root / DEFAULT_PROJECT_SETTINGS["config_session_path"]
    draft_path = root / DEFAULT_PROJECT_SETTINGS["editable_config_draft_path"]
    if any(
        path.exists()
        for path in (
            session_path,
            draft_path,
            root / "config_session.json",
            root / "search_config.project.json",
            root / "config/config_session.json",
            root / "config/search_config.project.json",
        )
    ):
        raise FileExistsError("该目录已有配置会话或设置 JSON；请使用“添加已有项目”，不会覆盖原文件")
    from phase_agent.configuration.session.project_scaffold import create_project_scaffold

    create_project_scaffold(root, execution_mode=execution_mode)
    register_project(target, registry=registry)
    return target


def describe_project(path: Path) -> str:
    config = _load_config(path)
    if config.get("service_start_policy", "manual") != "manual":
        raise ValueError("项目服务仅支持手动启动和停止")
    root = path.resolve().parent
    state = _load_optional_json(_resolve(root, config["state_path"]))
    session_path = _resolve(root, config.get("config_session_path", "config_session.json"))
    session = _load_optional_json(session_path)
    memory = state.get("decision_memory") or {}
    long_term = memory.get("long_term") or {}
    recent = memory.get("short_term") or {}
    workspace = ((session.get("config") or {}).get("storage") or {}).get("workspace_root") or root
    tasks = state.get("tasks") or []
    tasks = list(tasks.values()) if isinstance(tasks, dict) else tasks
    counts = {}
    for task in tasks:
        status = task.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    return (
        f"工作区：{workspace}\n"
        f"配置：{'已确认' if session.get('status') == 'confirmed' else '待确认'}\n"
        f"模式：{'自动模式' if config.get('execution_mode') == 'automatic' else '审批模式'}\n"
        f"任务：{counts or '尚未生成'}\n"
        "服务：用户手动启动和停止；更新不会自动启动。\n"
        f"项目长期记忆：{'有（人工审核）' if long_term else '无'}\n"
        f"项目短期记忆：{'有（运行状态）' if recent else '无'}\n"
        "记忆由本项目审核管理；Studio只提供对话与调试入口。"
    )


def _load_config(path: Path) -> dict:
    from phase_agent.runtime.runtime_config_io import _load_json_object

    data = _load_json_object(path, "运行时配置")
    for key in ("state_path", "ledger_path"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise ValueError(f"运行时配置 {key} 必须是非空路径字符串；文件：{path}")
    return data


def _resolve(root: Path, path: str) -> Path:
    value = Path(path)
    return (value if value.is_absolute() else root / value).resolve()


def _data_paths(path: Path) -> tuple[Path, Path]:
    config = _load_config(path)
    root = path.resolve().parent
    return _resolve(root, config["state_path"]), _resolve(root, config["ledger_path"])


def _load_optional_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    from phase_agent.runtime.runtime_config_io import _load_json_object

    return _load_json_object(path, "项目记录")


def _port_is_free(port: int) -> bool:
    with socket.socket() as connection:
        return connection.connect_ex(("127.0.0.1", port)) != 0


def _stop_owned_process(process, *, timeout: float = 5) -> bool:
    """Stop an owned or verified project service and all its child processes."""
    if process is None or not _process_is_running(process):
        return True
    from phase_agent.runtime.studio_run_lifecycle import interrupt_owned_studio
    import psutil

    interrupt_owned_studio(process)
    children = []
    if type(getattr(process, "pid", None)) is int:
        try:
            children = psutil.Process(process.pid).children(recursive=True)
        except psutil.NoSuchProcess:
            return True
    for child in reversed(children):
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    try:
        process.terminate()
        process.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, psutil.TimeoutExpired):
        process.kill()
        try:
            process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, psutil.TimeoutExpired):
            return False
    except (OSError, psutil.NoSuchProcess):
        pass
    _, alive = psutil.wait_procs(children, timeout=timeout)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    _, remaining = psutil.wait_procs(alive, timeout=timeout)
    return not remaining and not _process_is_running(process)


def stop_project_service(path, process=None):
    """Explicit Stop may attach only to the selected project's verified service."""
    from phase_agent.runtime.project_service import read_service, write_service
    import psutil

    if process is None:
        service = read_service(path, require_ready=False)
        if service is None:
            return True
        process = psutil.Process(service["pid"])
        command = process.cmdline()
        if "phase_agent.runtime.studio_service" not in command or "--runtime-config" not in command:
            raise RuntimeError("服务进程身份不匹配，未停止")
        configured = Path(command[command.index("--runtime-config") + 1]).resolve()
        if configured != Path(path).resolve():
            raise RuntimeError("服务不属于所选项目，未停止")
    service = read_service(path, require_ready=False)
    if service:
        write_service(path, {**service, "status": "stopping"})
    stopped = _stop_owned_process(process)
    if stopped:
        write_service(path, {"status": "stopped"})
    return stopped


def _existing_agent(port: int, selected: Path):
    """Identify an occupied port by process command line; fail closed if ambiguous."""
    try:
        import psutil

        listeners = [
            item
            for item in psutil.net_connections(kind="tcp")
            if item.status == "LISTEN"
            and item.pid
            and item.laddr.port == port
            and item.laddr.ip in {"127.0.0.1", "::1"}
        ]
        if len(listeners) != 1:
            return None
        process = psutil.Process(listeners[0].pid)
        command = process.cmdline()
        if not any(
            arg in {"phase_agent.runtime.agent_api", "phase_agent/runtime/agent_api.py"}
            for arg in command
        ):
            return None
        if "--runtime-config" in command:
            configured = command[command.index("--runtime-config") + 1]
        else:
            configured = next(
                (arg.split("=", 1)[1] for arg in command if arg.startswith("--runtime-config=")),
                None,
            )
        if not configured or str(Path(configured).resolve()).casefold() != str(selected).casefold():
            return None
        return process
    except (OSError, IndexError, ValueError, ImportError):
        return None


def _agent_accepts_token(port: int, token: str) -> bool:
    request = Request(
        f"http://127.0.0.1:{port}/v1/models", headers={"Authorization": f"Bearer {token}"}
    )
    try:
        with urlopen(request, timeout=2) as response:
            return response.status == 200
    except OSError:
        return False


def _agent_config_protocol_current(port: int) -> bool:
    try:
        with urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
            health = json.load(response)
        return health.get("config_protocol") == 2
    except (OSError, ValueError, TypeError):
        return False


def _process_is_running(process) -> bool:
    if process is None:
        return False
    if callable(getattr(process, "poll", None)):
        return process.poll() is None
    return process.is_running()


def _check_start_cancelled(cancel_event, process=None):
    if cancel_event is not None and cancel_event.is_set():
        if process is not None and not _stop_owned_process(process):
            raise RuntimeError("启动已取消，但服务仍在退出；请再次停止")
        raise RuntimeError("启动已取消；服务保持停止")


def restart_agent(path: Path, process=None, *, port: int | None = None, cancel_event=None):
    """Explicitly stop and relaunch one project, excluding concurrent launches."""
    from phase_agent.runtime.project_service import project_lock

    _check_start_cancelled(cancel_event)
    with project_lock(path, lock_name="launch.lock"):
        _check_start_cancelled(cancel_event)
        if not stop_project_service(path, process):
            raise RuntimeError("原服务未能停止，未启动新服务；请稍后重试")
        _check_start_cancelled(cancel_event)
        return _start_agent(path, port=port, cancel_event=cancel_event)


def start_agent(
    path: Path, *, port: int | None = None, cancel_event=None
) -> tuple[object, str, bool]:
    from phase_agent.runtime.project_service import project_lock

    _check_start_cancelled(cancel_event)
    with project_lock(path, lock_name="launch.lock"):
        return _start_agent(path, port=port, cancel_event=cancel_event)


def _start_agent(
    path: Path, *, port: int | None = None, cancel_event=None
) -> tuple[object, str, bool]:
    """Start an isolated project service, or open its verified existing instance."""
    from phase_agent.runtime.project_service import available_port, read_service

    path = path.resolve()
    _check_start_cancelled(cancel_event)
    config = _load_config(path)
    if config.get("service_start_policy", "manual") != "manual":
        raise ValueError("项目服务仅支持手动启动和停止")
    if path.is_file():
        validate_project_paths(path)
    existing = read_service(path)
    if existing:
        import psutil

        process = psutil.Process(existing["pid"])
        process.studio_port = existing["studio_port"]
        process.control_port = existing["control_port"]
        return process, local_service_tokens()[0], True
    owner = read_service(path, require_ready=False)
    if owner:
        raise RuntimeError(
            f"该项目服务仍存在（{owner.get('status', 'unhealthy')}），请先停止后重启；未重复启动"
        )
    from phase_agent.runtime.startup_diagnostics import startup_timeout, failure_summary

    timeout = startup_timeout(config)
    port = port if port is not None else available_port(config.get("control_port", 8765))
    if not _port_is_free(port):
        raise RuntimeError(
            f"端口 {port} 已被占用。Agent 不再接回后台进程；请关闭原启动窗口或结束旧 Agent 后重试。"
        )
    tool_token, control_token = local_service_tokens()
    env = os.environ.copy()
    env["OPENWEBUI_TOOL_TOKEN"] = tool_token
    env["OPENWEBUI_CONTROL_TOKEN"] = control_token
    ui_port = available_port(config.get("studio_port", 2024), excluded={port})
    if type(ui_port) is not int or not 1 <= ui_port <= 65535 or ui_port == port:
        raise ValueError("agent_web_port 必须是与控制端口不同的有效端口")
    if not _port_is_free(ui_port):
        raise RuntimeError(f"聊天端口 {ui_port} 已被占用；请先停止原服务")
    env["PHASE_WEB_USERNAME"] = "phase"
    env["PHASE_WEB_PASSWORD"] = tool_token
    deadline = time.monotonic() + timeout
    command = [
        sys.executable,
        "-u",
        "-m",
        "phase_agent.runtime.studio_service",
        "--runtime-config",
        str(path),
        "--control-port",
        str(port),
        "--port",
        str(ui_port),
        "--parent-pid",
        str(os.getpid()),
        "--startup-deadline",
        str(deadline),
    ]
    from datetime import datetime, timezone

    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log_path = path.parent / "logs" / f"agent_start_{attempt}.log"
    env["PHASE_STARTUP_LOG"] = str(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("a", encoding="utf-8")
    try:
        _check_start_cancelled(cancel_event)
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    finally:
        stream.close()
    while time.monotonic() < deadline:
        _check_start_cancelled(cancel_event, process)
        if process.poll() is not None:
            raise RuntimeError(
                f"Agent 启动失败（退出码 {process.poll()}）\n{failure_summary(path, log_path)}"
            )
        try:
            with urlopen(f"http://127.0.0.1:{port}/health", timeout=0.5) as response:
                if response.status == 200:
                    request = Request(f"http://127.0.0.1:{ui_port}/info")
                    with urlopen(request, timeout=0.5) as ui_response:
                        if ui_response.status == 200:
                            _check_start_cancelled(cancel_event, process)
                            process.studio_port = ui_port
                            process.control_port = port
                            if type(process.pid) is int:
                                import psutil
                                from phase_agent.runtime.project_service import write_service

                                write_service(
                                    path,
                                    {
                                        "pid": process.pid,
                                        "process_started": psutil.Process(
                                            process.pid
                                        ).create_time(),
                                        "runtime_config": str(path),
                                        "project_name": path.parent.name,
                                        "control_port": port,
                                        "studio_port": ui_port,
                                        "status": "ready",
                                    },
                                )
                            return process, tool_token, False
        except OSError:
            time.sleep(0.2)
    stopped = _stop_owned_process(process)
    outcome = "已停止本次启动" if stopped else "进程仍在退出，请先停止后再启动"
    raise RuntimeError(
        f"Agent 未能在 {timeout:g} 秒内就绪；{outcome}\n{failure_summary(path, log_path)}"
    )


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, help="Open one project's runtime config")
    parser.add_argument(
        "--check", action="store_true", help="Read-only diagnostics; never starts service"
    )
    args = parser.parse_args(argv)
    if args.check:
        if args.project is None:
            parser.error("--check requires --project")
        from phase_agent.runtime.project_diagnostics import inspect_runtime

        report = inspect_runtime(args.project)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["status"] in {"ready", "configuration_required"} else 2
    from phase_agent.runtime.project_dashboard import main as dashboard

    dashboard(initial_project=args.project)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

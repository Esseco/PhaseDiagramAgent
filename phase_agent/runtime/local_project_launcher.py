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


PROJECT_CONFIG_NAME = "agent_runtime.json"
DEFAULT_PROJECT_SETTINGS = {
    "agent_web_port": 7932,
    "config_session_path": "parameters/config_session.json",
    "editable_config_draft_path": "parameters/search_config.project.json",
    "state_path": "workflow_state/state.json",
    "ledger_path": "workflow_state/ledgers/phase_data.json",
    "new_runs_directory": "open_webui_runs",
    "phase_diagram_directory": "analysis_outputs",
    "approval_directory": "workflow_state/approvals",
    "local_action_directory": "workflow_state/approved_batches",
    "execution_mode": "debug",
    "run_steps_per_click": 1,
    "resume_existing_project": True,
    "deepseek": {
        "model": "deepseek-flash",
        "base_url": "https://api.deepseek.com",
        "max_tokens": 2400,
        "timeout": 60,
        "configuration_thinking": "disabled",
        "thinking": "enabled",
    },
    "manual_upload": {"enabled": False, "submit": False},
    "runtime_adapters": {},
}


def registry_path() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    if not base:
        raise RuntimeError("找不到本机 LOCALAPPDATA；请指定项目配置文件手动启动")
    return Path(base) / "PhaseSearchAgent" / "projects.json"


def read_project_registry(path: Path) -> list[str]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or any(not isinstance(row, str) for row in data):
        raise ValueError(f"项目清单格式无效：{path}")
    return data


def register_project(path: Path, *, registry: Path | None = None) -> list[str]:
    """Register a config without moving any state or overwriting existing projects."""
    registry = registry or registry_path()
    target = path.resolve()
    if not target.is_file():
        raise FileNotFoundError(f"运行时配置不存在：{target}")
    new_state, new_ledger = _data_paths(target)
    projects = read_project_registry(registry)
    for old in projects:
        old_path = Path(old)
        if old_path.resolve() != target and old_path.is_file():
            old_state, old_ledger = _data_paths(old_path)
            if old_state == new_state or old_ledger == new_ledger:
                raise ValueError("两个项目指向同一 state 或台账；请为新项目选择独立工作区")
            if target.parent == old_path.resolve().parent:
                raise ValueError("每个文件夹只能注册一个项目配置")
    validate_project_paths(target)
    if str(target) not in projects:
        projects.append(str(target))
        registry.parent.mkdir(parents=True, exist_ok=True)
        temporary = registry.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(projects, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(registry)
    return projects


def validate_project_paths(path):
    """Writable runtime and confirmed science storage must stay in this folder."""
    path = Path(path).resolve()
    root = path.parent
    settings = _load_config(path)
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
            if not target.is_relative_to(root):
                raise ValueError(f"项目写入路径 {key} 超出项目文件夹：{target}")
    session = _load_optional_json(
        _resolve(root, settings.get("config_session_path", "parameters/config_session.json"))
    )
    for config in (
        session.get("config") or {},
        (session.get("confirmed_snapshot") or {}).get("config") or {},
        {"storage": settings.get("runtime_storage_override") or {}},
    ):
        if settings.get("local_path_relocations"):
            from phase_agent.configuration.session.relocate_workspace_paths import (
                relocate_workspace_paths,
            )

            config = relocate_workspace_paths(config, settings["local_path_relocations"], root)
        storage = config.get("storage") or {}
        if storage.get("workspace_root"):
            owner = _resolve(root, storage["workspace_root"])
            if owner != root:
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
    try:
        with target.open("x", encoding="utf-8") as stream:
            settings = dict(DEFAULT_PROJECT_SETTINGS)
            settings.update(
                execution_mode=execution_mode,
                run_steps_per_click=1 if execution_mode == "debug" else 10,
            )
            json.dump(settings, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except FileExistsError:
        raise FileExistsError(f"该工作区已有 {target.name}；请选择“添加已有项目”，不会覆盖它")
    from phase_agent.configuration.defaults.default_layered_search_config import (
        default_layered_search_config,
    )
    from phase_agent.configuration.session.apply_config_revision import apply_config_revision
    from phase_agent.configuration.session.create_config_draft import create_config_draft
    from phase_agent.configuration.session.project_config_json import create_project_config_json
    from phase_agent.configuration.session.resolve_workspace_paths import default_workspace_storage
    from phase_agent.configuration.session.save_config_session import save_config_session
    from phase_agent.runtime.configuration_chat import BOOTSTRAP_HINTS

    config = default_layered_search_config()
    # New projects must not inherit the previous chemistry or an HPC model path.
    config["mlip"]["model_path"] = None
    config["system"]["constraints"]["TM_ratio"] = {}
    storage = default_workspace_storage(root)
    session = create_config_draft(config, require_workspace_path=True)
    session["bootstrap_hints"] = BOOTSTRAP_HINTS
    create_project_config_json(
        draft_path,
        config,
        bootstrap_hints={
            **BOOTSTRAP_HINTS,
            "local_initial_structure_directory": str(root / "structures" / "reference_structures"),
        },
        workspace_defaults=storage,
    )
    from phase_agent.configuration.session.split_project_config import split_project_file

    split_project_file(draft_path)
    session = apply_config_revision(
        session,
        {"storage": storage},
        reasons={"storage": "用户在本地启动器中选择了此独立项目工作区。"},
        author="user_selected_project_workspace",
    )
    session["setup_stage"] = "json_ready"
    session["editable_config_json_path"] = str(draft_path)
    save_config_session(session, session_path)
    from phase_agent.analysis.feedback.workspace_guide import publish_workspace_guide

    publish_workspace_guide(root)
    register_project(target, registry=registry)
    return target


def describe_project(path: Path) -> str:
    config = _load_config(path)
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
        f"项目长期记忆：{'有（人工审核）' if long_term else '无'}\n"
        f"项目短期记忆：{'有（运行状态）' if recent else '无'}\n"
        "记忆由本项目审核管理；Studio只提供对话与调试入口。"
    )


def _load_config(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(
        data.get(key) for key in ("state_path", "ledger_path")
    ):
        raise ValueError("运行时配置至少需要 state_path 和 ledger_path")
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
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"文件不是 JSON 对象：{path}")
    return data


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
    psutil.wait_procs(alive, timeout=timeout)
    return not _process_is_running(process)


def stop_project_service(path, process=None):
    """Explicit Stop may attach only to the selected project's verified service."""
    from phase_agent.runtime.project_service import read_service, write_service
    import psutil

    if process is None:
        service = read_service(path)
        if service is None:
            return True
        process = psutil.Process(service["pid"])
        command = process.cmdline()
        if "phase_agent.runtime.studio_service" not in command or "--runtime-config" not in command:
            raise RuntimeError("服务进程身份不匹配，未停止")
        configured = Path(command[command.index("--runtime-config") + 1]).resolve()
        if configured != Path(path).resolve():
            raise RuntimeError("服务不属于所选项目，未停止")
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


def start_agent(path: Path, *, port: int | None = None) -> tuple[object, str, bool]:
    """Start an isolated project service, or open its verified existing instance."""
    from phase_agent.runtime.project_service import available_port, read_service

    path = path.resolve()
    config = _load_config(path)
    if path.is_file():
        validate_project_paths(path)
    existing = read_service(path)
    if existing:
        import psutil

        process = psutil.Process(existing["pid"])
        process.studio_port = existing["studio_port"]
        process.control_port = existing["control_port"]
        return process, local_service_tokens()[0], True
    timeout = config.get("agent_startup_timeout_seconds", 45)
    if type(timeout) not in {int, float} or not 5 <= timeout <= 600:
        raise ValueError("agent_startup_timeout_seconds 必须在 5–600 秒之间")
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
    ]
    log_path = path.parent / "logs" / "agent_server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("a", encoding="utf-8")
    try:
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
    deadline = time.monotonic() + float(timeout)
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Agent 启动失败；请查看 {log_path}")
        try:
            with urlopen(f"http://127.0.0.1:{port}/health", timeout=0.5) as response:
                if response.status == 200:
                    request = Request(f"http://127.0.0.1:{ui_port}/info")
                    with urlopen(request, timeout=0.5) as ui_response:
                        if ui_response.status == 200:
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
    _stop_owned_process(process)
    raise RuntimeError(f"Agent 未能在 {timeout:g} 秒内就绪；已停止本次启动，请查看 {log_path}")


def main() -> None:
    from phase_agent.runtime.project_dashboard import main as dashboard

    dashboard()


if __name__ == "__main__":
    main()

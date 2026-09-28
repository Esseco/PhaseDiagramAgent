"""Small local project picker and one-click launcher for the Open WebUI Agent."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import webbrowser

from run.local_service_tokens import local_service_tokens
from run.open_webui_startup import effective_webui_config, load_startup_settings


PROJECT_CONFIG_NAME = "open_webui_runtime.json"
DEFAULT_PROJECT_SETTINGS = {
    "open_webui_url": "http://127.0.0.1:3000",
    "config_session_path": "config_session.json",
    "editable_config_draft_path": "search_config.project.json",
    "state_path": "current/state.json",
    "ledger_path": "current/phase_data.json",
    "new_runs_directory": "open_webui_runs",
    "phase_diagram_directory": "current/phase_diagrams",
    "approval_directory": "current/approvals",
    "local_action_directory": "current/approved_batches",
    "execution_mode": "debug",
    "deepseek": {"model": "deepseek-flash", "base_url": "https://api.deepseek.com",
                 "max_tokens": 2400, "timeout": 60, "configuration_thinking": "disabled",
                 "thinking": "enabled"},
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
    if str(target) not in projects:
        projects.append(str(target))
        registry.parent.mkdir(parents=True, exist_ok=True)
        temporary = registry.with_suffix(".tmp")
        temporary.write_text(json.dumps(projects, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(registry)
    return projects


def create_project(workspace_root: Path, *, registry: Path | None = None) -> Path:
    """Create an isolated runtime and editable science draft after GUI path selection."""
    from config_layer.session.validate_workspace_root import validate_workspace_root
    validate_workspace_root(str(workspace_root))
    root = workspace_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / PROJECT_CONFIG_NAME
    session_path = root / "config_session.json"
    draft_path = root / "search_config.project.json"
    if session_path.exists() or draft_path.exists():
        raise FileExistsError("该目录已有配置会话或设置 JSON；请使用“添加已有项目”，不会覆盖原文件")
    try:
        with target.open("x", encoding="utf-8") as stream:
            settings = {**DEFAULT_PROJECT_SETTINGS, **load_startup_settings()}
            json.dump(settings, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except FileExistsError:
        raise FileExistsError(f"该工作区已有 {target.name}；请选择“添加已有项目”，不会覆盖它")
    from config_layer.defaults.default_layered_search_config import default_layered_search_config
    from config_layer.session.apply_config_revision import apply_config_revision
    from config_layer.session.create_config_draft import create_config_draft
    from config_layer.session.project_config_json import create_project_config_json
    from config_layer.session.resolve_workspace_paths import default_workspace_storage
    from config_layer.session.save_config_session import save_config_session
    from run.configuration_chat import BOOTSTRAP_HINTS
    config = default_layered_search_config()
    storage = default_workspace_storage(root)
    session = create_config_draft(config, require_workspace_path=True)
    session["bootstrap_hints"] = BOOTSTRAP_HINTS
    create_project_config_json(draft_path, config, bootstrap_hints=BOOTSTRAP_HINTS,
                                workspace_defaults=storage)
    session = apply_config_revision(
        session, {"storage": storage},
        reasons={"storage": "用户在本地启动器中选择了此独立项目工作区。"},
        author="user_selected_project_workspace",
    )
    session["setup_stage"] = "json_ready"
    session["editable_config_json_path"] = str(draft_path)
    save_config_session(session, session_path)
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
    return (f"工作区：{workspace}\n"
            f"配置：{'已确认' if session.get('status') == 'confirmed' else '待确认'}\n"
            f"项目长期记忆：{'有（人工审核）' if long_term else '无'}\n"
            f"项目短期记忆：{'有（运行状态）' if recent else '无'}\n"
            "Open WebUI 账户记忆：由 Open WebUI 单独管理，可能跨聊天和项目；"
            "建议关闭 phase-search-agent 模型的 Memory 注入。")


def _load_config(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(data.get(key) for key in ("state_path", "ledger_path")):
        raise ValueError("运行时配置至少需要 state_path 和 ledger_path")
    return data


def _open_webui_url(config: dict) -> str:
    url = config.get("open_webui_url", "http://127.0.0.1:3000")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("open_webui_url 必须是有效的 HTTP(S) 网页地址")
    return url


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


def _webui_is_available(url: str) -> bool:
    try:
        with urlopen(url, timeout=2) as response:
            return response.status < 500
    except OSError:
        return False


def _ensure_webui(config: dict, workspace: Path, url: str, *, on_started=None) -> tuple[bool, subprocess.Popen | None]:
    """Return readiness and the owned process, if this call started Open WebUI."""
    if _webui_is_available(url):
        return True, None
    command = config.get("open_webui_start_command")
    if command is None:
        return False, None
    if not isinstance(command, list) or not command or any(not isinstance(arg, str) or not arg for arg in command):
        raise ValueError("open_webui_start_command 必须是非空命令参数列表，不使用 shell")
    configured_environment = config.get("open_webui_environment") or {}
    if (not isinstance(configured_environment, dict)
            or any(not isinstance(key, str) or not isinstance(value, str)
                   for key, value in configured_environment.items())):
        raise ValueError("open_webui_environment 必须是字符串到字符串的 JSON 对象")
    workdir = Path(config.get("open_webui_workdir") or workspace).expanduser().resolve()
    if not workdir.is_dir():
        raise ValueError(f"Open WebUI 工作目录不存在：{workdir}")
    log_path = workspace / "open_webui_server.log"
    environment = os.environ.copy()
    environment.update(configured_environment)
    timeout = config.get("open_webui_startup_timeout_seconds", 120)
    if type(timeout) not in {int, float} or not 5 <= timeout <= 600:
        raise ValueError("open_webui_startup_timeout_seconds 必须在 5–600 秒之间")
    with log_path.open("a", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=workdir, env=environment,
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if on_started is not None:
        on_started(process)
    deadline = time.monotonic() + float(timeout)
    while time.monotonic() < deadline:
        if _webui_is_available(url):
            return True, process if process.poll() is None else None
        exit_code = process.poll()
        if exit_code not in (None, 0):
            raise RuntimeError(f"Open WebUI 启动命令退出（代码 {exit_code}）；请查看 {log_path}")
        time.sleep(0.5)
    _stop_owned_process(process)
    raise RuntimeError(f"Open WebUI 尚未在 {timeout:g} 秒内就绪；请查看 {log_path}")


def _stop_owned_process(process, *, timeout: float = 5) -> bool:
    """Stop a process started by this launcher; never look up or kill by port/name."""
    if process is None:
        return True
    if process.poll() is not None:
        return True
    try:
        process.terminate()
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
    except OSError:
        return process.poll() is not None
    return process.poll() is not None


def _existing_agent(port: int, selected: Path):
    """Identify an occupied port by process command line; fail closed if ambiguous."""
    try:
        import psutil
        listeners = [item for item in psutil.net_connections(kind="tcp")
                     if item.status == "LISTEN" and item.pid and item.laddr.port == port
                     and item.laddr.ip in {"127.0.0.1", "::1"}]
        if len(listeners) != 1:
            return None
        process = psutil.Process(listeners[0].pid)
        command = process.cmdline()
        if not any(arg in {"run.open_webui_api", "run/open_webui_api.py"} for arg in command):
            return None
        if "--runtime-config" in command:
            configured = command[command.index("--runtime-config") + 1]
        else:
            configured = next((arg.split("=", 1)[1] for arg in command
                               if arg.startswith("--runtime-config=")), None)
        if not configured or str(Path(configured).resolve()).casefold() != str(selected).casefold():
            return None
        return process
    except (OSError, IndexError, ValueError, ImportError):
        return None


def _agent_accepts_token(port: int, token: str) -> bool:
    request = Request(f"http://127.0.0.1:{port}/v1/models",
                      headers={"Authorization": f"Bearer {token}"})
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
    if isinstance(process, subprocess.Popen):
        return process.poll() is None
    return process.is_running()


def start_agent(path: Path, *, port: int = 8765) -> tuple[object, str, bool]:
    """Start one launcher-owned Agent; never reconnect to a background process."""
    path = path.resolve()
    config = _load_config(path)
    timeout = config.get("agent_startup_timeout_seconds", 45)
    if type(timeout) not in {int, float} or not 5 <= timeout <= 600:
        raise ValueError("agent_startup_timeout_seconds 必须在 5–600 秒之间")
    if not _port_is_free(port):
        raise RuntimeError(
            f"端口 {port} 已被占用。Agent 不再接回后台进程；请关闭原启动窗口或结束旧 Agent 后重试。"
        )
    tool_token, control_token = local_service_tokens()
    env = os.environ.copy()
    env["OPENWEBUI_TOOL_TOKEN"] = tool_token
    env["OPENWEBUI_CONTROL_TOKEN"] = control_token
    command = [sys.executable, "-u", "-m", "run.open_webui_api", "--runtime-config", str(path),
               "--port", str(port), "--parent-pid", str(os.getpid())]
    log_path = path.parent / "agent_server.log"
    stream = log_path.open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(command, cwd=Path(__file__).resolve().parent.parent,
                                   env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    finally:
        stream.close()
    deadline = time.monotonic() + float(timeout)
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Agent 启动失败；请查看 {log_path}")
        try:
            with urlopen(f"http://127.0.0.1:{port}/health", timeout=0.5) as response:
                if response.status == 200:
                    return process, tool_token, False
        except OSError:
            time.sleep(0.2)
    process.terminate()
    raise RuntimeError(f"Agent 未能在 {timeout:g} 秒内就绪；已停止本次启动，请查看 {log_path}")


def main() -> None:
    registry = registry_path()
    app = tk.Tk()
    app.title("相图搜索 Agent · 项目启动")
    app.geometry("760x390")
    selected = tk.StringVar()
    details = tk.StringVar(value="请选择项目；新聊天不会自动建立新项目。")
    running = {"process": None, "webui_process": None, "owned": False,
               "starting_webui": False, "closing": False}
    projects = [item for item in read_project_registry(registry) if Path(item).is_file()]
    ttk.Label(app, text="选择项目（每个项目有独立工作区、state 和台账）").pack(pady=10)
    chooser = ttk.Combobox(app, textvariable=selected, values=projects, state="readonly", width=100)
    chooser.pack(padx=15)
    ttk.Label(app, textvariable=details, justify="left", wraplength=720).pack(padx=18, pady=18)

    def refresh() -> None:
        if selected.get():
            try:
                details.set(describe_project(Path(selected.get())))
            except (OSError, ValueError, KeyError) as error:
                details.set(f"项目配置需要检查：{error}")

    def add_project() -> None:
        path = filedialog.askopenfilename(title="选择已有项目的 open_webui_runtime.json",
                                         filetypes=[("JSON", "*.json")])
        if path:
            try:
                projects[:] = register_project(Path(path), registry=registry)
                chooser.configure(values=projects)
                selected.set(str(Path(path).resolve()))
                refresh()
            except (OSError, ValueError, KeyError) as error:
                messagebox.showerror("无法添加项目", str(error))

    def new_project() -> None:
        root = filedialog.askdirectory(title="选择新项目工作区根目录")
        if root:
            try:
                path = create_project(Path(root), registry=registry)
                projects[:] = read_project_registry(registry)
                chooser.configure(values=projects)
                selected.set(str(path))
                refresh()
                messagebox.showinfo("配置文件已生成", f"请编辑 {path.parent / 'search_config.project.json'}；\n"
                                    "然后在 Agent 对话中发送“读取配置 JSON 并继续”。")
            except (OSError, ValueError) as error:
                messagebox.showerror("无法新建项目", str(error))

    def launch() -> None:
        if running["starting_webui"]:
            return
        if not selected.get():
            messagebox.showinfo("请选择项目", "先选择继续的项目，或新建独立项目。")
            return
        try:
            project_config = effective_webui_config(_load_config(Path(selected.get())))
            open_webui_url = _open_webui_url(project_config)
            process, tool_token, reused = start_agent(Path(selected.get()))
            running["process"] = process
            running["owned"] = True
            app.clipboard_clear()
            app.clipboard_append(tool_token)
            details.set(describe_project(Path(selected.get())) +
                        "\nAgent 已启动，并绑定到当前启动窗口；关闭窗口将停止 Agent 及由本启动器启动的 Open WebUI。"
                        "Open WebUI 连接密钥已复制到剪贴板；仅首次添加连接时粘贴。")
            local_chat_url = "http://127.0.0.1:8765/phase/chat"
            if not _webui_is_available(open_webui_url) and not project_config.get("open_webui_start_command"):
                details.set(details.get() + "\nOpen WebUI 未运行；已打开本地对话页。连接密钥已复制，点“从剪贴板连接”即可对话。")
                webbrowser.open(local_chat_url)
                return
            running["starting_webui"] = True
            details.set(details.get() + "\n正在启动 Open WebUI；窗口仍可操作，请稍候……")

            def finish_webui(ready, process, error):
                running["starting_webui"] = False
                if running["closing"]:
                    if process is not None:
                        _stop_owned_process(process)
                    return
                running["webui_process"] = process
                if error is not None:
                    details.set(details.get() + f"\nOpen WebUI 启动失败：{error}。已打开本地对话页。")
                    webbrowser.open(local_chat_url)
                elif ready:
                    webbrowser.open(open_webui_url)
                else:
                    details.set(details.get() + "\nOpen WebUI 未响应；已打开本地对话页。")
                    webbrowser.open(local_chat_url)

            def start_webui_in_background():
                try:
                    ready, process = _ensure_webui(
                        project_config, Path(selected.get()).resolve().parent,
                        open_webui_url,
                        on_started=lambda started: running.__setitem__("webui_process", started),
                    )
                    error = None
                except (OSError, ValueError, RuntimeError) as failure:
                    ready, process, error = False, None, failure
                app.after(0, finish_webui, ready, process, error)

            threading.Thread(target=start_webui_in_background, daemon=True).start()
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            if running["owned"]:
                stop(show_warning=False)
            messagebox.showerror("启动失败", str(error))

    def stop(*, show_warning: bool = True) -> bool:
        process = running["process"]
        if not _stop_owned_process(process):
            if show_warning:
                messagebox.showwarning("服务仍在退出", "Agent 未能停止；请检查 agent_server.log")
            return False
        webui_process = running["webui_process"]
        if not _stop_owned_process(webui_process):
            if show_warning:
                messagebox.showwarning(
                    "服务仍在退出",
                    "由本启动器启动的 Open WebUI 未能停止；请稍后重试，日志文件可能仍被占用。",
                )
            return False
        running["process"] = None
        running["webui_process"] = None
        running["owned"] = False
        refresh()
        return True

    def close() -> None:
        running["closing"] = True
        if running["owned"]:
            if not stop():
                return
        app.destroy()

    row = ttk.Frame(app)
    row.pack(pady=4)
    ttk.Button(row, text="继续所选项目", command=launch).pack(side="left", padx=5)
    ttk.Button(row, text="新建独立项目", command=new_project).pack(side="left", padx=5)
    ttk.Button(row, text="添加已有项目", command=add_project).pack(side="left", padx=5)
    ttk.Button(row, text="停止本次 Agent", command=stop).pack(side="left", padx=5)
    chooser.bind("<<ComboboxSelected>>", lambda _event: refresh())
    if projects:
        selected.set(projects[-1])
        refresh()
    app.protocol("WM_DELETE_WINDOW", close)
    app.mainloop()


if __name__ == "__main__":
    main()

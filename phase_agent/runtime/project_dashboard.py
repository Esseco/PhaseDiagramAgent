"""Folder-first desktop project manager. All workers are project-scoped."""

import json
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from urllib.request import Request, urlopen

from phase_agent.runtime.local_project_launcher import (
    PROJECT_CONFIG_NAME,
    create_project,
    describe_project,
    read_project_registry,
    register_project,
    registry_path,
    start_agent,
    _stop_owned_process,
    _load_config,
    _load_optional_json,
    _resolve,
)
from phase_agent.runtime.project_service import read_service
from phase_agent.runtime.project_service import service_directory


def select_folder(folder, registry, *, execution_mode="debug"):
    root = Path(folder).resolve()
    config = root / PROJECT_CONFIG_NAME
    if config.is_file():
        register_project(config, registry=registry)
    else:
        config = create_project(root, registry=registry, execution_mode=execution_mode)
    return config


def choose_initial_mode(parent):
    dialog = tk.Toplevel(parent)
    dialog.title("新项目 · 运行模式")
    dialog.transient(parent)
    dialog.resizable(False, False)
    mode = tk.StringVar(dialog, value="debug")
    result = []
    ttk.Label(dialog, text="选择运行模式，保存后用于此项目：").pack(
        anchor="w", padx=20, pady=(16, 8)
    )
    ttk.Radiobutton(
        dialog, text="审批模式（默认）：先给建议，你批准后执行", variable=mode, value="debug"
    ).pack(anchor="w", padx=20, pady=6)
    ttk.Radiobutton(
        dialog,
        text="自动模式：Agent 决定后自动推进，无需逐步批准",
        variable=mode,
        value="automatic",
    ).pack(anchor="w", padx=20, pady=6)
    controls = ttk.Frame(dialog)
    controls.pack(fill="x", padx=20, pady=16)

    def confirm():
        result.append(mode.get())
        dialog.destroy()

    ttk.Button(controls, text="建立项目", command=confirm).pack(side="right")
    ttk.Button(controls, text="取消", command=dialog.destroy).pack(side="right", padx=8)
    dialog.grab_set()
    parent.wait_window(dialog)
    return result[0] if result else None


def set_project_mode(config, mode):
    if mode not in {"automatic", "debug"}:
        raise ValueError("未知运行模式")
    if read_service(config):
        raise ValueError("请先停止此项目服务，再更改运行模式")
    settings = _load_config(config)
    settings.update(execution_mode=mode, resume_existing_project=True)
    settings.setdefault("run_steps_per_click", 10)
    temporary = config.with_suffix(".tmp")
    temporary.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(config)


def send_start(config, service, token):
    settings = _load_config(config)
    session = _load_optional_json(
        _resolve(
            config.parent, settings.get("config_session_path", "parameters/config_session.json")
        )
    )
    if session.get("status") != "confirmed":
        return "首次配置尚未确认，请在 Studio 中设置体系、母结构和计算参数后开始。"
    base = f"http://127.0.0.1:{service['studio_port']}"

    def api(route, payload):
        request = Request(
            base + route,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=900) as response:
            return json.load(response)

    record = service_directory(config) / "launcher_thread.json"
    if record.is_file():
        thread = json.loads(record.read_text(encoding="utf-8"))["thread_id"]
    else:
        thread = api(
            "/threads",
            {"metadata": {"project_name": config.parent.name, "source": "project_launcher"}},
        )["thread_id"]
        record.write_text(json.dumps({"thread_id": thread}), encoding="utf-8")
    import uuid

    value = api(
        f"/threads/{thread}/runs/wait",
        {
            "assistant_id": "phase_chat",
            "input": {
                "messages": [{"role": "user", "content": "开始搜索", "id": str(uuid.uuid4())}]
            },
        },
    )
    if value.get("__error__"):
        raise RuntimeError(str(value["__error__"]))
    return value["messages"][-1]["content"]


def main():
    registry = registry_path()
    app = tk.Tk()
    app.title("相图搜索 · 项目管理")
    app.geometry("980x680")
    app.minsize(720, 480)
    style = ttk.Style(app)
    style.configure("Treeview", rowheight=30)
    ttk.Label(app, text="相图搜索项目", font=("Microsoft YaHei UI", 18, "bold")).pack(
        anchor="w", padx=20, pady=(16, 4)
    )
    ttk.Label(
        app, text="选择文件夹并启动服务，复制下方 Studio 网址到浏览器。计算操作在 Studio 中完成。"
    ).pack(anchor="w", padx=20, pady=(0, 12))
    buttons = ttk.Frame(app)
    buttons.pack(fill="x", padx=20, pady=(0, 12))
    tree = ttk.Treeview(
        app, columns=("name", "service", "config", "folder"), show="headings", height=7
    )
    for column, title, width in (
        ("name", "项目", 160),
        ("service", "服务", 95),
        ("config", "配置", 95),
        ("folder", "文件夹", 530),
    ):
        tree.heading(column, text=title)
        tree.column(column, width=width)
    tree.pack(fill="x", padx=20)
    text = tk.Text(app, height=13, wrap="word", font=("Microsoft YaHei UI", 10), state="disabled")
    text.pack(fill="both", expand=True, padx=20, pady=12)
    owned, busy, replies = {}, set(), queue.Queue()

    def display(value):
        text.configure(state="normal")
        text.delete("1.0", "end")
        text.insert("1.0", value)
        text.configure(state="disabled")

    def selected():
        items = tree.selection()
        return Path(items[0]) if items else None

    def refresh():
        chosen = selected()
        remaining = set(tree.get_children())
        for value in read_project_registry(registry):
            path = Path(value)
            if not path.is_file():
                continue
            try:
                settings = _load_config(path)
                session = _load_optional_json(
                    _resolve(
                        path.parent,
                        settings.get("config_session_path", "parameters/config_session.json"),
                    )
                )
                status = "操作中" if path in busy else "运行中" if read_service(path) else "已停止"
                values = (
                    path.parent.name,
                    status,
                    "已确认" if session.get("status") == "confirmed" else "待配置",
                    str(path.parent),
                )
            except (OSError, ValueError):
                values = (path.parent.name, "配置异常", "需检查", str(path.parent))
            if tree.exists(str(path)):
                tree.item(str(path), values=values)
            else:
                tree.insert("", "end", iid=str(path), values=values)
            remaining.discard(str(path))
        for item in remaining:
            tree.delete(item)
        if chosen and tree.exists(str(chosen)):
            if str(chosen) not in tree.selection():
                tree.selection_set(str(chosen))

    def add():
        folder = filedialog.askdirectory(title="选择项目文件夹（已有项目自动加载）")
        if folder:
            try:
                mode = "debug"
                if not (Path(folder) / PROJECT_CONFIG_NAME).is_file():
                    mode = choose_initial_mode(app)
                    if mode is None:
                        return
                path = select_folder(folder, registry, execution_mode=mode)
                refresh()
                tree.selection_set(str(path))
                show()
            except (OSError, ValueError) as error:
                messagebox.showerror("无法加载项目", str(error))

    def show(_event=None):
        path = selected()
        if path:
            try:
                service = read_service(path)
                url = (
                    f"\n\nStudio 网址：\nhttps://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:{service['studio_port']}"
                    if service
                    else "\n\n服务尚未启动。"
                )
                display(f"项目：{path.parent.name}\n" + describe_project(path) + url)
            except (OSError, ValueError, KeyError) as error:
                display(str(error))

    def launch():
        path = selected()
        if path is None or path in busy:
            return
        busy.add(path)
        display(f"正在启动 {path.parent.name}…")
        refresh()

        def worker():
            try:
                process, token, reused = start_agent(path)
                service = {"studio_port": process.studio_port, "control_port": process.control_port}
                replies.put(("ready", path, (process, reused, service)))
            except Exception as error:
                replies.put(
                    (
                        "message",
                        path,
                        f"操作失败：{error}\n查看 {path.parent / 'logs/agent_server.log'}",
                    )
                )
            finally:
                replies.put(("done", path, None))

        threading.Thread(target=worker, daemon=True).start()

    def stop():
        path = selected()
        process = owned.get(path)
        from phase_agent.runtime.local_project_launcher import stop_project_service

        if stop_project_service(path, process):
            owned.pop(path, None)
            display("本地服务已停止；已提交的超算作业可能继续运行。重启后先同步作业状态。")
        refresh()

    def drain():
        while not replies.empty():
            kind, path, value = replies.get_nowait()
            if kind == "ready":
                process, reused, service = value
                if not reused:
                    owned[path] = process
                if selected() == path:
                    show()
            elif kind == "message" and selected() == path:
                display(value)
            elif kind == "done":
                busy.discard(path)
                refresh()
        app.after(200, drain)

    def close():
        if busy:
            messagebox.showinfo(
                "操作进行中", "请等待启动或计算请求返回后再关闭；其他项目仍可操作。"
            )
            return
        if owned and not messagebox.askokcancel(
            "关闭项目管理", "将停止本窗口启动的本地服务；超算作业不会自动取消。"
        ):
            return
        for process in owned.values():
            if not _stop_owned_process(process):
                messagebox.showerror("退出失败", "服务仍在退出，请稍后重试。")
                return
        app.destroy()

    def open_process():
        path = selected()
        service = read_service(path) if path else None
        if not service:
            display("请先启动所选项目服务，再查看本轮过程。")
            return
        import webbrowser

        webbrowser.open(f"http://127.0.0.1:{service['studio_port']}/phase/process")

    def manage_conversations():
        path = selected()
        service = read_service(path) if path else None
        if not service:
            display("请先启动所选项目服务。")
            return
        import webbrowser

        webbrowser.open(f"http://127.0.0.1:{service['studio_port']}/phase/conversations")

    for column, (label, command) in enumerate(
        (
            ("选择文件夹", add),
            ("启动服务", launch),
            ("查看状态", show),
            ("查看本轮过程", open_process),
            ("管理对话", manage_conversations),
            ("停止服务", stop),
        )
    ):
        ttk.Button(buttons, text=label, command=command).grid(
            row=0, column=column, sticky="ew", padx=3, pady=4
        )
        buttons.columnconfigure(column, weight=1)

    def auto_refresh():
        try:
            refresh()
        except (OSError, ValueError) as error:
            display(f"项目状态读取失败：{error}")
        app.after(3000, auto_refresh)

    tree.bind("<<TreeviewSelect>>", show)
    tree.bind("<Double-1>", show)
    refresh()
    app.after(200, drain)
    app.after(3000, auto_refresh)
    app.protocol("WM_DELETE_WINDOW", close)
    app.mainloop()


if __name__ == "__main__":
    main()

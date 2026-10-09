import json
import tkinter as tk

import pytest
from run import project_dashboard


def test_ui_builds_folder_project_controls(tmp_path, monkeypatch):
    monkeypatch.setattr(project_dashboard, "registry_path", lambda: tmp_path / "projects.json")
    original = tk.Tk
    labels = []
    def window():
        root = original()
        root.withdraw()
        def inspect():
            root.deiconify()
            root.geometry("720x480")
            root.update()
            def visit(widget):
                if "text" in widget.keys():
                    label = str(widget.cget("text"))
                    labels.append(label)
                    if label in {"选择文件夹", "启动服务", "查看状态", "停止服务"}:
                        assert widget.winfo_ismapped()
                        assert 0 <= widget.winfo_rooty() - root.winfo_rooty() < 200
                        assert widget.winfo_rootx() + widget.winfo_width() <= root.winfo_rootx() + root.winfo_width()
                for child in widget.winfo_children():
                    visit(child)
            visit(root)
            root.destroy()
        root.mainloop = inspect
        return root
    monkeypatch.setattr(tk, "Tk", window)
    project_dashboard.main()
    for name in ("选择文件夹", "启动服务", "查看状态", "停止服务"):
        assert name in labels
    assert "更多" not in labels
    assert "刷新" not in labels
    assert "设置 / 密钥" not in labels


def test_mode_switch_preserves_other_project_settings(tmp_path, monkeypatch):
    config = tmp_path / "agent_runtime.json"
    config.write_text(json.dumps({"state_path": "state.json", "ledger_path": "ledger.json",
                                  "deepseek": {"model": "existing"}}))
    monkeypatch.setattr(project_dashboard, "read_service", lambda _: None)
    project_dashboard.set_project_mode(config, "automatic")
    value = json.loads(config.read_text())
    assert value["deepseek"]["model"] == "existing"
    assert value["resume_existing_project"] is True
    assert value["run_steps_per_click"] == 10
    monkeypatch.setattr(project_dashboard, "read_service", lambda _: {"status": "ready"})
    with pytest.raises(ValueError, match="先停止"):
        project_dashboard.set_project_mode(config, "debug")

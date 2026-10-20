from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
from threading import Event
from types import SimpleNamespace

import pytest

from phase_agent.runtime import local_project_launcher as launcher


def fake_lifecycle(monkeypatch):
    calls = []

    @contextmanager
    def lock(path, *, lock_name):
        calls.append(("lock", path, lock_name))
        yield
        calls.append(("unlock", path))

    monkeypatch.setattr("phase_agent.runtime.project_service.project_lock", lock)
    monkeypatch.setattr(
        launcher,
        "stop_project_service",
        lambda path, process: calls.append(("stop", path, process)) or True,
    )
    monkeypatch.setattr(
        launcher,
        "_start_agent",
        lambda path, **kwargs: calls.append(("start", path, kwargs)) or ("new", "token", False),
    )
    return calls


def test_restart_stops_before_start_under_same_project_lock(monkeypatch):
    calls = fake_lifecycle(monkeypatch)
    path = Path("selected/agent_runtime.json")
    process, cancel = object(), Event()
    assert launcher.restart_agent(path, process, port=2024, cancel_event=cancel) == (
        "new",
        "token",
        False,
    )
    assert calls == [
        ("lock", path, "launch.lock"),
        ("stop", path, process),
        ("start", path, {"port": 2024, "cancel_event": cancel}),
        ("unlock", path),
    ]


def test_restart_can_stop_a_verified_service_from_another_launcher(monkeypatch):
    calls = fake_lifecycle(monkeypatch)
    path = Path("selected/agent_runtime.json")
    launcher.restart_agent(path)
    assert calls[1] == ("stop", path, None)


def test_restart_does_not_start_when_old_process_will_not_exit(monkeypatch):
    calls = fake_lifecycle(monkeypatch)
    monkeypatch.setattr(launcher, "stop_project_service", lambda *args: False)
    with pytest.raises(RuntimeError, match="未能停止"):
        launcher.restart_agent(Path("selected/agent_runtime.json"))
    assert not any(call[0] == "start" for call in calls)


def test_restart_preserves_identity_errors_and_does_not_launch(monkeypatch):
    calls = fake_lifecycle(monkeypatch)

    def reject(*args):
        raise RuntimeError("服务不属于所选项目")

    monkeypatch.setattr(launcher, "stop_project_service", reject)
    with pytest.raises(RuntimeError, match="不属于所选项目"):
        launcher.restart_agent(Path("selected/agent_runtime.json"))
    assert not any(call[0] == "start" for call in calls)


@pytest.mark.parametrize("cancel_during_stop", [False, True])
def test_cancelled_restart_never_launches(monkeypatch, cancel_during_stop):
    calls = fake_lifecycle(monkeypatch)
    cancel = Event()
    if cancel_during_stop:

        def stop(*args):
            calls.append(("stop",))
            cancel.set()
            return True

        monkeypatch.setattr(launcher, "stop_project_service", stop)
    else:
        cancel.set()
    with pytest.raises(RuntimeError, match="启动已取消"):
        launcher.restart_agent(Path("selected/agent_runtime.json"), cancel_event=cancel)
    assert not any(call[0] == "start" for call in calls)
    assert any(call[0] == "stop" for call in calls) == cancel_during_stop


def test_dashboard_restart_button_uses_worker_and_ignores_duplicate_clicks(tmp_path):
    # Keep the native Tcl interpreter's lifecycle isolated from other UI tests.
    code = (
        "from pathlib import Path; from pytest import MonkeyPatch; import sys; "
        "from tests.test_project_service_restart import _dashboard_restart_scenario; "
        "m=MonkeyPatch(); "
        "_dashboard_restart_scenario(Path(sys.argv[1]), m); m.undo()"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=25,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _dashboard_restart_scenario(tmp_path, monkeypatch):
    from phase_agent.runtime import project_dashboard as dashboard

    path = tmp_path / "agent_runtime.json"
    path.write_text(
        json.dumps({"state_path": "state.json", "ledger_path": "ledger.json"}), encoding="utf-8"
    )
    registry = tmp_path / "projects.json"
    registry.write_text(json.dumps([str(path)]), encoding="utf-8")
    monkeypatch.setattr(dashboard, "registry_path", lambda: registry)
    monkeypatch.setattr(dashboard, "read_service", lambda _: {"studio_port": 2024})
    monkeypatch.setattr(dashboard, "describe_project", lambda _: "测试项目")
    workers, calls, scheduled, stopped = [], [], [], []
    monkeypatch.setattr(
        launcher,
        "stop_project_service",
        lambda config, process: stopped.append((config, process)) or True,
    )

    class Thread:
        def __init__(self, *, target, daemon):
            assert daemon
            self.target = target

        def start(self):
            workers.append(self.target)

    def restart(config, process=None, *, cancel_event=None):
        calls.append((config, process, cancel_event))
        if len(calls) == 1:
            raise RuntimeError("原服务未能停止")
        return SimpleNamespace(studio_port=2024, control_port=8765), "token", False

    monkeypatch.setattr(dashboard.threading, "Thread", Thread)
    monkeypatch.setattr(dashboard, "restart_agent", restart, raising=False)
    monkeypatch.setattr(
        dashboard, "start_agent", lambda *a, **kw: pytest.fail("Restart used ordinary start")
    )
    original = dashboard.tk.Tk

    def window():
        root = original()
        root.withdraw()
        root.after = lambda delay, callback: scheduled.append(callback)

        def inspect():
            widgets = []

            def visit(widget):
                widgets.append(widget)
                for child in widget.winfo_children():
                    visit(child)

            visit(root)
            tree = next(widget for widget in widgets if isinstance(widget, dashboard.ttk.Treeview))
            button = next(
                (
                    widget
                    for widget in widgets
                    if isinstance(widget, dashboard.ttk.Button)
                    and widget.cget("text") == "重启服务"
                ),
                None,
            )
            assert button is not None
            tree.selection_set(str(path))
            text = next(widget for widget in widgets if isinstance(widget, dashboard.tk.Text))
            for index, expected in enumerate(("重启失败：原服务未能停止", "2024")):
                button.invoke()
                button.invoke()
                assert len(workers) == index + 1 and len(calls) == index
                assert "操作中" in tree.item(str(path))["values"]
                workers[index]()
                assert calls[index][0] == path and calls[index][1] is None
                assert isinstance(calls[index][2], Event)
                next(callback for callback in scheduled if callback.__name__ == "drain")()
                assert expected in text.get("1.0", "end")
            button.invoke()
            assert len(workers) == 3
            workers[2]()
            stop = next(
                widget
                for widget in widgets
                if isinstance(widget, dashboard.ttk.Button) and widget.cget("text") == "停止服务"
            )
            stop.invoke()
            assert stopped == [(path, None)]
            assert calls[2][2].is_set()
            next(callback for callback in scheduled if callback.__name__ == "drain")()
            root.destroy()

        root.mainloop = inspect
        return root

    monkeypatch.setattr(dashboard.tk, "Tk", window)
    dashboard.main()

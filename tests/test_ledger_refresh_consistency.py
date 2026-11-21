from phase_agent.persistence.ledger.phase_data_manager import PhaseDataManager, refresh_if_changed
import pytest


def test_refresh_detects_unsaved_changes(tmp_path):
    path = tmp_path / "ledger.json"
    manager = PhaseDataManager(
        {"P": ["O3"], "H": {"O3": [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]}, "TM_ratio": {"Ni": 1}}
    )
    manager.save(path)
    other = PhaseDataManager.load(path)
    other.data["external"] = 1
    other.save(path)
    manager.data["local"] = 2
    with pytest.raises(ValueError, match="ledger_memory_changed"):
        refresh_if_changed(manager, path)
    assert manager.data["local"] == 2


def test_refresh_reads_external_update(tmp_path):
    path = tmp_path / "ledger.json"
    manager = PhaseDataManager(
        {"P": ["O3"], "H": {"O3": [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]}, "TM_ratio": {"Ni": 1}}
    )
    manager.save(path)
    other = PhaseDataManager.load(path)
    other.data["external"] = 1
    other.save(path)
    assert refresh_if_changed(manager, path)
    assert manager.data["external"] == 1
    assert not refresh_if_changed(manager, path)


def test_refresh_observes_separate_process(tmp_path):
    import subprocess
    import sys

    path = tmp_path / "ledger.json"
    manager = PhaseDataManager(
        {"P": ["O3"], "H": {"O3": [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]}, "TM_ratio": {"Ni": 1}}
    )
    manager.save(path)
    script = "from phase_agent.persistence.ledger.phase_data_manager import PhaseDataManager; import sys; m=PhaseDataManager.load(sys.argv[1]); m.data['external']=123; m.save(sys.argv[1])"
    subprocess.run([sys.executable, "-c", script, str(path)], check=True)
    assert refresh_if_changed(manager, path)
    assert manager.data["external"] == 123


def test_stale_writer_must_not_overwrite_external_update(tmp_path):
    path = tmp_path / "ledger.json"
    manager = PhaseDataManager(
        {"P": ["O3"], "H": {"O3": [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]}, "TM_ratio": {"Ni": 1}}
    )
    manager.save(path)
    stale = PhaseDataManager.load(path)
    current = PhaseDataManager.load(path)
    current.data["external"] = 1
    current.save(path)
    stale.data["local"] = 2
    with pytest.raises(ValueError, match="ledger.*changed"):
        stale.save(path)
    assert PhaseDataManager.load(path).data["external"] == 1


def test_context_cache_reuses_and_invalidates_sources(tmp_path, monkeypatch):
    import json
    from threading import RLock
    from types import SimpleNamespace
    from phase_agent.runtime.local_agent_control import LocalAgentControl
    import phase_agent.runtime.codex.context as context_module

    state = tmp_path / "state.json"
    config = tmp_path / "config.json"
    ledger = tmp_path / "ledger.json"
    state.write_text("{}", encoding="utf-8")
    config.write_text("{}", encoding="utf-8")
    manager = PhaseDataManager(
        {"P": ["O3"], "H": {"O3": [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]}, "TM_ratio": {"Ni": 1}}
    )
    manager.save(ledger)
    handler = SimpleNamespace(
        state_path=state,
        lock=RLock(),
        config_session_path=config,
        workflow_kwargs={"manager": manager, "run_config": {"ledger_path": str(ledger)}},
    )
    control = LocalAgentControl(handler)
    calls = []
    monkeypatch.setattr(
        context_module, "build_codex_context", lambda _: calls.append(1) or {"count": len(calls)}
    )
    assert control.codex_context()["count"] == 1
    assert control.codex_context()["count"] == 1
    manager.data["in_memory"] = True
    assert control.codex_context()["count"] == 2
    config.write_text(json.dumps({"changed": True}), encoding="utf-8")
    assert control.codex_context()["count"] == 3
    other = PhaseDataManager.load(ledger)
    other.data["external"] = 1
    other.save(ledger)
    with pytest.raises(ValueError, match="ledger_memory_changed"):
        control.codex_context()
    assert manager.data["in_memory"] is True


def test_two_process_writers_do_not_overwrite(tmp_path):
    import subprocess
    import sys
    import time

    path = tmp_path / "ledger.json"
    manager = PhaseDataManager(
        {"P": ["O3"], "H": {"O3": [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]}, "TM_ratio": {"Ni": 1}}
    )
    manager.save(path)
    script = """import sys,time
from pathlib import Path
from phase_agent.persistence.ledger.phase_data_manager import PhaseDataManager
p,ready,go,value=map(Path,sys.argv[1:])
m=PhaseDataManager.load(p); m.data["writer"]=value.name; ready.write_text("ready")
while not go.exists(): time.sleep(0.01)
try: m.save(p); print("saved")
except ValueError as e: print(str(e))
"""
    go = tmp_path / "go"
    procs = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                script,
                str(path),
                str(tmp_path / f"ready{i}"),
                str(go),
                str(tmp_path / str(i)),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for i in range(2)
    ]
    try:
        deadline = time.monotonic() + 30
        while not all((tmp_path / f"ready{i}").exists() for i in range(2)):
            if time.monotonic() > deadline:
                raise TimeoutError("writers did not reach barrier")
            time.sleep(0.02)
        go.write_text("go")
        outputs = [p.communicate(timeout=30) for p in procs]
        assert all(p.returncode == 0 for p in procs), outputs
        assert sorted(out.strip() for out, _ in outputs) == ["ledger_changed_since_load", "saved"]
        assert PhaseDataManager.load(path).data["writer"] in {"0", "1"}
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


def test_unbound_manager_cannot_replace_existing_ledger(tmp_path):
    path = tmp_path / "ledger.json"
    boundary = {
        "P": ["O3"],
        "H": {"O3": [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]},
        "TM_ratio": {"Ni": 1},
    }
    original = PhaseDataManager(boundary)
    original.save(path)
    unbound = PhaseDataManager(boundary)
    unbound.data["other"] = True
    with pytest.raises(ValueError, match="ledger_existing_target_requires_load"):
        unbound.save(path)
    assert "other" not in PhaseDataManager.load(path).data


def test_workflow_refresh_detects_ledger_change_without_new_control_revision(tmp_path):
    from phase_agent.persistence.result_recovery_commit import refresh_manager_after_recovery

    path = tmp_path / "ledger.json"
    boundary = {
        "P": ["O3"],
        "H": {"O3": [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]},
        "TM_ratio": {"Ni": 1},
    }
    manager = PhaseDataManager(boundary)
    manager.save(path)
    other = PhaseDataManager.load(path)
    other.data["external"] = "new"
    other.save(path)
    assert refresh_manager_after_recovery(manager, {}, path)
    assert manager.data["external"] == "new"

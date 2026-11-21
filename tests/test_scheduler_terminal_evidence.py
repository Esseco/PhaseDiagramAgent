from copy import deepcopy

import pytest

from phase_agent.tools.state.scheduler_terminal_evidence import (
    bind_terminal_observations, terminal_reservation_is_proven,
)


def fixture():
    task = {"task_id": "DFT-1", "task_key": "key", "task_checksum": "sha",
            "config_version": "config", "stage": "dft_single_point", "status": "cancelled",
            "batch_id": "remote-01", "input_path": r"E:\project\inputs\batch_remote-01\00000-DFT-1\task.json"}
    observation = {"task_id": "DFT-1", "job_id": "123", "state": "CANCELLED",
                   "workdir": "/work/inputs/batch_remote-01/00000-DFT-1",
                   "elapsed": "01:00:00", "observed_at": "2026-11-09T01:00:00Z",
                   "source": "electerm_sacct"}
    reservation = {"stage": "dft_single_point", "status": "reserved",
                   "reserved_cost": 10, "config_version": "config"}
    return {"tasks": [task], "budget_reservations": {"key": reservation}}, observation


def test_terminal_evidence_preserves_task_and_unsettled_cost():
    state, row = fixture()
    updated = bind_terminal_observations(state, [row])
    assert "scheduler_terminal_evidence" not in state["tasks"][0]
    assert updated["budget_reservations"] == state["budget_reservations"]
    assert updated["tasks"][0]["status"] == "cancelled"
    assert terminal_reservation_is_proven(updated, "key", updated["budget_reservations"]["key"])
    assert bind_terminal_observations(updated, [row]) == updated


@pytest.mark.parametrize("field,value", [("state", "RUNNING"), ("workdir", "/other/00000-DFT-1"),
    ("job_id", ""), ("source", "wait_waiver"), ("task_id", "other")])
def test_unbound_or_nonterminal_evidence_is_rejected(field, value):
    state, row = fixture()
    row[field] = value
    before = deepcopy(state)
    with pytest.raises(ValueError):
        bind_terminal_observations(state, [row])
    assert state == before


def test_local_waiver_or_changed_task_never_proves_remote_terminal():
    state, row = fixture()
    state["tasks"][0]["recovery_wait_waived"] = True
    assert not terminal_reservation_is_proven(state, "key", state["budget_reservations"]["key"])
    state = bind_terminal_observations(state, [row])
    state["tasks"][0]["task_checksum"] = "changed"
    assert not terminal_reservation_is_proven(state, "key", state["budget_reservations"]["key"])


def test_formal_control_records_idempotently(tmp_path):
    from threading import RLock
    from types import SimpleNamespace
    from phase_agent.runtime.local_agent_control import LocalAgentControl
    from phase_agent.tools.step_runner.file_protocol import read_json, write_json

    state, row = fixture()
    path = tmp_path / "state.json"
    write_json(path, state)
    control = LocalAgentControl(SimpleNamespace(state_path=path, lock=RLock()))
    result = control.register_scheduler_terminal_evidence([row])
    assert result["status"] == "recorded" and not result["costs_settled"]
    saved = read_json(path, {})
    assert terminal_reservation_is_proven(saved, "key", saved["budget_reservations"]["key"])
    assert control.register_scheduler_terminal_evidence([row])["status"] == "reused"


def test_reused_evidence_rejects_concurrent_state_change(tmp_path, monkeypatch):
    import subprocess
    import sys
    from threading import RLock
    from types import SimpleNamespace
    import phase_agent.tools.state.scheduler_terminal_evidence as module
    from phase_agent.runtime.local_agent_control import LocalAgentControl
    from phase_agent.tools.step_runner.file_protocol import read_json, write_json

    state, row = fixture()
    path = tmp_path / "state.json"
    write_json(path, bind_terminal_observations(state, [row]))
    original = module.bind_terminal_observations

    def competing(current, observations):
        result = original(current, observations)
        script = ("import sys; from phase_agent.tools.step_runner.file_protocol import read_json, write_json; "
                  "p=sys.argv[1]; s=read_json(p); s['other_process_update']=True; write_json(p,s)")
        subprocess.run([sys.executable, "-c", script, str(path)], check=True)
        return result

    monkeypatch.setattr(module, "bind_terminal_observations", competing)
    control = LocalAgentControl(SimpleNamespace(state_path=path, lock=RLock()))
    with pytest.raises(ValueError, match="stale_state_file_version"):
        control.register_scheduler_terminal_evidence([row])
    assert read_json(path)["other_process_update"] is True

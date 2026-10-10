import pytest
from phase_agent.tools.workflows import run_event_loop as module


@pytest.mark.parametrize("status", ["pending", "running", "submitted", "unknown"])
def test_formal_waiting_task_stops_next_action(tmp_path, monkeypatch, status):
    calls = []
    def fake_step(**kwargs):
        calls.append(kwargs)
        return {"status": "completed", "state": {**kwargs["state"], "tasks": [
            {"task_id": "t", "stage": "dft_single_point", "status": status}]}}
    monkeypatch.setattr(module, "run_tool_step", fake_step)
    result = module.run_event_loop({}, {}, registry={}, max_steps=3,
                                   state_path=tmp_path / "state.json")
    assert len(calls) == 1
    assert result["status"] == "tasks_in_progress"


def test_explicit_waiver_does_not_reopen_wait(tmp_path, monkeypatch):
    calls = []
    def fake_step(**kwargs):
        calls.append(kwargs)
        return {"status": "completed", "state": {**kwargs["state"], "tasks": [
            {"task_id": "t", "stage": "dft_single_point", "status": "unknown",
             "recovery_wait_waived": True}]}}
    monkeypatch.setattr(module, "run_tool_step", fake_step)
    module.run_event_loop({}, {}, registry={}, max_steps=2, state_path=tmp_path / "state.json")
    assert len(calls) == 2

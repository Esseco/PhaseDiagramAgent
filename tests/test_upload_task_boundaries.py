from copy import deepcopy

import pytest

from phase_agent.tools.remote.batch_runner import RemoteBatchRunner


def test_dft_preparation_ignores_completed_business_summaries(tmp_path):
    summary = {"task_id": "turn:task", "task_key": "generate_branches:c:s", "status": "completed"}
    task = {
        "task_id": "D1",
        "task_key": "dft:1",
        "stage": "dft_single_point",
        "status": "pending",
        "structure_id": "s3",
    }
    state = {
        "tasks": [summary, task],
        "generation_history": [
            {"registered_ids": ["s1"]},
            {"registered_ids": ["s2"]},
            {"registered_ids": ["s3"]},
        ],
        "budget_reservations": {"dft:1": {"status": "reserved"}},
    }
    original = deepcopy(state)
    dispatched = []
    runner = RemoteBatchRunner(
        tmp_path / "inputs",
        worker_command=[],
        dispatcher=lambda row: dispatched.append(row["task_id"]) or {"generator": "atomate"},
        stage_profiles={"dft": {"stages": ["dft_single_point"], "vasp_command": "vasp_std"}},
    )
    result = runner.prepare(state)
    assert dispatched == ["D1"]
    assert result["batch"]["task_ids"] == ["D1"]
    assert result["state"]["tasks"][0] == summary
    assert result["state"]["tasks"][1]["search_group_index"] == 3
    assert state == original


def test_real_pending_calculation_still_requires_unambiguous_origin(tmp_path):
    state = {
        "tasks": [
            {"task_id": "D1", "task_key": "dft:1", "stage": "dft_single_point", "status": "pending"}
        ],
        "generation_history": [{}, {}, {}],
        "budget_reservations": {"dft:1": {"status": "reserved"}},
    }
    with pytest.raises(ValueError, match="无法唯一确定"):
        RemoteBatchRunner(tmp_path, worker_command=[]).prepare(state)
    assert not list(tmp_path.iterdir())


def test_pending_business_summary_is_never_a_remote_task(tmp_path):
    state = {
        "tasks": [{"task_id": "turn:task", "task_key": "summary", "status": "pending"}],
        "generation_history": [{}, {}, {}],
        "budget_reservations": {"summary": {"status": "reserved"}},
    }
    assert RemoteBatchRunner(tmp_path, worker_command=[]).prepare(state)["status"] == "no_tasks"

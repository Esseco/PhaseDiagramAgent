from copy import deepcopy

import pytest

from phase_agent.tools.remote.batch_runner import RemoteBatchRunner
from phase_agent.tools.local.prepare_model_refresh import RefreshUploadRunner


def scoped_state():
    return {
        "tasks": [
            {"task_id": "bad-mc", "task_key": "mc", "stage": "deep_search", "status": "pending"},
            {
                "task_id": "D1",
                "task_key": "dft",
                "stage": "dft_single_point",
                "status": "pending",
                "structure_id": "s3",
            },
        ],
        "generation_history": [
            {"registered_ids": ["s1"]},
            {"registered_ids": ["s2"]},
            {"registered_ids": ["s3"]},
        ],
        "budget_reservations": {"mc": {"status": "reserved"}, "dft": {"status": "reserved"}},
    }


def test_other_pending_task_cannot_block_or_enter_approved_dft_preparation(tmp_path):
    state = scoped_state()
    before = deepcopy(state)
    runner = RemoteBatchRunner(
        tmp_path,
        worker_command=[],
        task_ids={"D1"},
        dispatcher=lambda row: {"generator": "atomate"},
        stage_profiles={"dft": {"stages": ["dft_single_point"], "vasp_command": "vasp_std"}},
    )
    result = runner.prepare(state)
    assert result["batch"]["task_ids"] == ["D1"]
    assert result["state"]["tasks"][0] == before["tasks"][0]
    assert state == before


def test_unreserved_task_is_not_validated_for_upload(tmp_path):
    state = scoped_state()
    state["budget_reservations"].pop("mc")
    runner = RemoteBatchRunner(
        tmp_path,
        worker_command=[],
        dispatcher=lambda row: {"generator": "atomate"},
        stage_profiles={"dft": {"stages": ["dft_single_point"], "vasp_command": "vasp_std"}},
    )
    assert runner.prepare(state)["batch"]["task_ids"] == ["D1"]


def test_explicit_empty_scope_prepares_nothing(tmp_path):
    runner = RemoteBatchRunner(tmp_path, worker_command=[], task_ids=set())
    assert runner.prepare(scoped_state())["status"] == "no_tasks"
    assert not list(tmp_path.iterdir())


def test_blocked_dedup_gate_does_not_validate_unrelated_compute_lineage(tmp_path):
    state = scoped_state()
    state["dedup_gate"] = {"status": "pending"}
    result = RemoteBatchRunner(tmp_path, worker_command=[]).prepare(state)
    assert result["status"] == "no_tasks" and result["state"]["tasks"] == state["tasks"]


def test_refresh_scope_is_applied_before_unrelated_provenance_validation(tmp_path):
    state = scoped_state()
    state["tasks"][1].update(model_refresh_id="refresh", model_refresh_wave=0)
    runner = RefreshUploadRunner(
        tmp_path,
        refresh_id="refresh",
        wave=0,
        worker_command=[],
        dispatcher=lambda row: {"generator": "atomate"},
    )
    # Read the parent's portable preparation only; real atomate/scripts are independently tested.
    result = RemoteBatchRunner.prepare(runner, state)
    assert result["batch"]["task_ids"] == ["D1"]
    assert result["state"]["tasks"][0] == state["tasks"][0]

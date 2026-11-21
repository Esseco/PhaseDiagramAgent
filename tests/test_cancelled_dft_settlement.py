from copy import deepcopy

import pytest

from phase_agent.tools.state.cancelled_dft_settlement import settle_cancelled


def state():
    task = {
        "task_id": "dft1", "task_key": "key", "stage": "dft_single_point",
        "status": "cancelled", "recovery_cancelled": True, "batch_id": "batch",
        "task_checksum": "hash", "cancellation_reason": "cancelled",
        "scheduler_terminal_evidence": {
            "state": "CANCELLED", "job_id": "123", "task_id": "dft1",
            "task_key": "key", "task_checksum": "hash", "batch_id": "batch",
        },
    }
    return {
        "tasks": [task], "budget_reservations": {
            "key": {"stage": "dft_single_point", "status": "reserved", "reserved_cost": 4},
            "pending": {"stage": "dft_single_point", "status": "reserved", "reserved_cost": 3},
        },
        "budget_usage": {"stages": {"dft_single_point": {"tasks": 100, "cost": 50}}},
        "reserved_relative_cost": 7,
    }


def test_terminal_settlement_counts_used_task_and_is_idempotent():
    original = state()
    updated, report = settle_cancelled(original, ["dft1"])
    assert report["settled_count"] == 1
    assert original == state()
    assert updated["budget_usage"]["stages"]["dft_single_point"]["tasks"] == 101
    assert updated["budget_reservations"]["key"]["actual_cost"] is None
    assert updated["budget_reservations"]["key"]["settlement_basis"] == "reserved_cost_upper_bound_actual_unknown"
    assert updated["budget_reservations"]["pending"] == original["budget_reservations"]["pending"]
    again, report = settle_cancelled(updated, ["dft1"])
    assert report["status"] == "already_settled" and again == updated


@pytest.mark.parametrize("field,value", [
    ("state", "RUNNING"), ("task_id", "other"), ("task_key", "other"),
    ("task_checksum", "other"), ("job_id", None), ("batch_id", "other"),
])
def test_wrong_scheduler_evidence_refuses_all_writes(field, value):
    s = state()
    s["tasks"][0]["scheduler_terminal_evidence"][field] = value
    before = deepcopy(s)
    with pytest.raises(ValueError, match="terminal_evidence_invalid"):
        settle_cancelled(s, ["dft1"])
    assert s == before


def test_conflicting_settlement_refuses():
    s = state()
    s["budget_reservations"]["key"].update(status="settled", task_status="completed")
    with pytest.raises(ValueError, match="settlement_conflict"):
        settle_cancelled(s, ["dft1"])

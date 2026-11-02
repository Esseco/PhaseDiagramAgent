from copy import deepcopy

import pytest

from phase_agent.graphs.project.runner import _business_state_is_clean_for_stale_checkpoint as clean
from phase_agent.tools.step_runner.file_protocol import write_json


def case():
    key = "model-refresh:plan:supplement-envelope"
    return {
        "confirmed_config_version": "config",
        "active_model_version": "new",
        "execution_recovery_report": {"status": "clear", "unsettled": []},
        "tasks": [{"task_id": "returned", "status": "completed"}],
        "model_refresh": {
            "new_model_version": "new",
            "status": "waiting_results",
            "wave": 0,
            "approved_by": "approved-refresh-plan",
            "supplemental_reservation_key": key,
            "plan": {
                "checksum": "plan",
                "new_model_version": "new",
                "maximum_supplemental_count": 1,
                "maximum_supplemental_cost": 10,
            },
            "approved_scope": {
                "plan_checksum": "plan",
                "config_version": "config",
                "model_version": "new",
                "maximum_count": 1,
                "maximum_cost": 10,
                "remaining_waves": 1,
                "tool": "prepare_local_batch_files",
                "submit_authorized": False,
            },
        },
        "budget_reservations": {
            key: {
                "task_key": key,
                "stage": "relax_and_feature",
                "config_version": "config",
                "model_version": "new",
                "reserved_cost": 10,
                "status": "reserved",
            }
        },
    }


def test_registered_planning_hold_does_not_mean_inflight_execution(tmp_path):
    state = case()
    before = deepcopy(state)
    path = tmp_path / "state.json"
    write_json(path, state)
    before_bytes = path.read_bytes()
    assert clean(path, state=state)
    assert state == before and path.read_bytes() == before_bytes


@pytest.mark.parametrize(
    "bad",
    [
        "different_key",
        "different_plan",
        "different_model",
        "different_config",
        "cost",
        "count",
        "unapproved",
        "scope_missing",
        "running_reservation",
        "unknown_reservation",
        "explicit_reconciliation",
        "pending_task",
        "real_recovery",
        "pending_recovery",
        "unrelated_hold",
    ],
)
def test_unproven_or_real_inflight_records_still_block(tmp_path, bad):
    state = case()
    refresh = state["model_refresh"]
    key = refresh["supplemental_reservation_key"]
    hold = state["budget_reservations"][key]
    if bad == "different_key":
        hold["task_key"] = "not-the-hold"
    elif bad == "different_plan":
        refresh["approved_scope"]["plan_checksum"] = "other"
    elif bad == "different_model":
        hold["model_version"] = "other"
    elif bad == "different_config":
        refresh["approved_scope"]["config_version"] = "other"
    elif bad == "cost":
        hold["reserved_cost"] = 11
    elif bad == "count":
        refresh["approved_scope"]["maximum_count"] = 2
    elif bad == "unapproved":
        refresh.pop("approved_by")
    elif bad == "scope_missing":
        refresh.pop("approved_scope")
    elif bad == "running_reservation":
        hold["status"] = "running"
    elif bad == "unknown_reservation":
        hold["status"] = "unknown"
    elif bad == "explicit_reconciliation":
        hold["reconciliation_required"] = True
    elif bad == "pending_task":
        state["tasks"][0]["status"] = "running"
    elif bad == "real_recovery":
        state["execution_recovery_report"] = {
            "status": "reconciliation_required",
            "unsettled": [{"tool": "prepare_local_batch_files"}],
        }
    elif bad == "pending_recovery":
        state["pending_execution_recoveries"] = {"review": {}}
    elif bad == "unrelated_hold":
        state["budget_reservations"]["another-action"] = {
            "status": "reserved",
            "config_version": "config",
        }
    path = tmp_path / "state.json"
    write_json(path, state)
    assert not clean(path, state=state)


def test_current_receipt_overrules_an_old_clear_report(tmp_path):
    from phase_agent.tools.state.execution_receipts import begin_execution

    state = case()
    path = tmp_path / "state.json"
    write_json(path, state)
    identity = {
        "invocation_id": "real-action",
        "config_version": "config",
        "action_hash": "hash",
        "tool": "prepare_local_batch_files",
    }
    assert begin_execution(path, identity)["allowed"]
    assert not clean(path, state=state)


def test_unreadable_current_receipts_remain_a_hard_boundary(tmp_path, monkeypatch):
    import sqlite3
    import phase_agent.tools.state.execution_receipts as receipts

    state = case()
    path = tmp_path / "state.json"
    write_json(path, state)

    def unreadable(*args):
        raise sqlite3.OperationalError("unreadable receipt")

    monkeypatch.setattr(receipts, "recovery_report", unreadable)
    assert not clean(path, state=state)

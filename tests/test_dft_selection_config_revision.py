"""Synthetic selection-only migrations never execute or rewrite existing work."""

from copy import deepcopy
import json

import pytest

from phase_agent.configuration.runtime.authorize_budget_extension import authorize_budget_extension


def revision():
    old = {
        "dft": {
            "selection": {
                "single_point_max_per_round": 32,
                "single_point_cost_per_round": 5000.0,
                "near_hull_ev_per_atom": 0.1,
            }
        },
        "mlip": {"version": "m1", "relax_parameters": {"fmax": 0.05}},
        "budgets": {
            "total_relative_cost": 20000,
            "stage_limits": {
                "dft_single_point": {"max_tasks": 100, "max_cost": 6000, "task_cost": 30},
            },
        },
        "mc_policy": {"second_segment_enabled": False},
    }
    new = deepcopy(old)
    new["dft"]["selection"].update(
        max_tasks_per_round=100,
        single_point_max_per_round=100,
        single_point_cost_per_round=None,
    )
    state = {
        "confirmed_config_version": "old",
        "confirmed_config": old,
        "tasks": [
            {
                "task_id": "mc",
                "stage": "deep_search",
                "status": "running",
                "config_version": "old",
                "model_version": "m1",
            },
            {
                "task_id": "dft",
                "stage": "dft_single_point",
                "status": "completed",
                "config_version": "old",
                "model_version": "m1",
                "energy": -7.25,
            },
        ],
        "active_model_version": "m1",
        "model_registry": {"m1": {"status": "active"}},
        "budget_limits": deepcopy(old["budgets"]),
        "budget_usage": {"total_relative_cost": 35},
        "budget_reservations": {"mc": {"status": "running", "reserved_cost": 4}},
        "pending_execution_policies": {
            "p": {
                "revision": 3,
                "project_identity": {"config_version": "old"},
                "agent_proposal": {
                    "recommended_action": "select_dft_candidates",
                    "raw_action": {"tool": "select_dft_candidates"},
                },
            }
        },
        "config_migrations": [{"type": "historical", "from": "initial", "to": "old"}],
        "phase_records": [{"record_id": "e", "energy": -7.25}],
        "current_state_snapshot": {"config_version": "old"},
    }
    return state, {"config_version": "new", "config": new}


@pytest.mark.parametrize(
    "status", ["pending", "running", "submitted", "queued", "reserved", "completed"]
)
def test_selection_revision_preserves_active_work(tmp_path, status):
    state, snapshot = revision()
    state["tasks"][0]["status"] = status
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    saved = path.read_bytes()
    original, confirmed = deepcopy(state), deepcopy(snapshot)

    result = authorize_budget_extension(state, snapshot, user_approved=True)

    assert result["status"] == "extended"
    migrated = result["state"]
    assert migrated["confirmed_config_version"] == "new"
    assert migrated["confirmed_config"] == snapshot["config"]
    assert migrated["config_migrations"][:-1] == state["config_migrations"]
    assert migrated["config_migrations"][-1]["type"] == "approved_dft_selection_policy_revision"
    for key in state.keys() - {"confirmed_config_version", "confirmed_config", "config_migrations"}:
        assert migrated[key] == state[key]
    assert len(result["policy_changes"]) == 3
    assert state == original and snapshot == confirmed
    assert path.read_bytes() == saved
    migrated["tasks"][0]["status"] = "failed"
    migrated["confirmed_config"]["dft"]["selection"]["max_tasks_per_round"] = 1
    assert state == original and snapshot == confirmed


@pytest.mark.parametrize("approved", [False, None, 0, 1, "yes"])
def test_selection_revision_requires_explicit_boolean_approval(approved):
    state, snapshot = revision()
    result = authorize_budget_extension(state, snapshot, user_approved=approved)
    assert result["status"] == "approval_required"
    assert result["state"] == state


@pytest.mark.parametrize(
    "path,value",
    [
        (("mlip", "version"), "m2"),
        (("mlip", "relax_parameters", "fmax"), 0.1),
        (("budgets", "total_relative_cost"), 30000),
        (("dft", "selection", "near_hull_ev_per_atom"), 0.2),
        (("mc_policy", "second_segment_enabled"), True),
    ],
)
def test_selection_revision_rejects_mixed_changes(path, value):
    state, snapshot = revision()
    cursor = snapshot["config"]
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    result = authorize_budget_extension(state, snapshot, user_approved=True)
    assert result["status"] == "rejected_non_budget_change"
    assert result["state"] == state
    assert ".".join(path) in result["changed_fields"]


def test_selection_revision_rejects_non_policy_type_changes():
    state, snapshot = revision()
    snapshot["config"]["mc_policy"]["second_segment_enabled"] = 0
    result = authorize_budget_extension(state, snapshot, user_approved=True)
    assert result["status"] == "rejected_non_budget_change"
    assert result["state"] == state


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_tasks_per_round", 99),
        ("max_tasks_per_round", 101),
        ("max_tasks_per_round", True),
        ("max_tasks_per_round", 100.0),
        ("single_point_max_per_round", 99),
        ("single_point_max_per_round", True),
        ("single_point_cost_per_round", 5000),
        ("single_point_cost_per_round", 0),
        ("single_point_cost_per_round", "missing"),
        ("max_tasks_per_round", "missing"),
    ],
)
def test_selection_revision_rejects_other_limits(field, value):
    state, snapshot = revision()
    selection = snapshot["config"]["dft"]["selection"]
    if value == "missing":
        selection.pop(field)
    else:
        selection[field] = value
    result = authorize_budget_extension(state, snapshot, user_approved=True)
    assert result["status"] == "rejected_invalid_dft_selection_policy"
    assert result["state"] == state


@pytest.mark.parametrize("legacy", [{}, {"selection": {}}])
def test_selection_revision_supports_missing_legacy_policy(legacy):
    state, snapshot = revision()
    state["confirmed_config"]["dft"] = legacy
    snapshot["config"] = deepcopy(state["confirmed_config"])
    snapshot["config"]["dft"]["selection"] = {
        "max_tasks_per_round": 100,
        "single_point_max_per_round": 100,
        "single_point_cost_per_round": None,
    }
    assert authorize_budget_extension(state, snapshot, user_approved=True)["status"] == "extended"


def test_selection_revision_requires_approval_even_for_empty_runs():
    state, snapshot = revision()
    state = {key: state[key] for key in ("confirmed_config_version", "confirmed_config")}
    result = authorize_budget_extension(state, snapshot)
    assert result["status"] == "approval_required"
    assert result["state"] == state


def test_selection_revision_is_idempotent():
    state, snapshot = revision()
    first = authorize_budget_extension(state, snapshot, user_approved=True)
    repeated = authorize_budget_extension(first["state"], snapshot, user_approved=True)
    assert repeated["status"] == "unchanged"
    assert repeated["state"] == first["state"]

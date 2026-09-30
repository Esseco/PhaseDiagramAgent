"""A confirmed second MC segment policy can rebind only an idle run."""

from copy import deepcopy

from config_layer.defaults.default_layered_search_config import default_layered_search_config
from config_layer.runtime.authorize_budget_extension import authorize_budget_extension


def _revision():
    old = default_layered_search_config()
    old["mc_policy"]["second_segment_enabled"] = False
    new = deepcopy(old)
    new["mc_policy"]["second_segment_enabled"] = True
    state = {
        "confirmed_config_version": "old", "confirmed_config": old,
        "tasks": [{"task_id": "first-mc", "stage": "deep_search",
                   "status": "completed", "config_version": "old"}],
        "budget_reservations": {"first-mc": {"status": "settled"}},
        "pending_execution_policies": {},
    }
    return state, {"config_version": "new", "config": new}


def test_second_mc_policy_revision_preserves_completed_task_version():
    state, snapshot = _revision()
    result = authorize_budget_extension(state, snapshot)

    assert result["status"] == "rebound"
    assert result["state"]["confirmed_config_version"] == "new"
    assert result["state"]["tasks"][0]["config_version"] == "old"
    assert result["state"]["config_migrations"][-1]["type"] == "confirmed_generation_policy_revision"
    assert state["confirmed_config_version"] == "old"


def test_second_mc_policy_revision_rejects_active_or_mixed_changes():
    state, snapshot = _revision()
    active = deepcopy(state)
    active["tasks"].append({"task_id": "pending", "status": "pending"})
    assert authorize_budget_extension(active, snapshot)["status"] == "rejected_active_work"

    mixed = deepcopy(snapshot)
    mixed["config"]["mlip"]["name"] = "different-model"
    assert authorize_budget_extension(state, mixed)["status"] == "approval_required"
    assert authorize_budget_extension(state, mixed, user_approved=True)["status"] == "rejected_non_budget_change"

from copy import deepcopy
import pytest
from phase_agent.configuration.runtime.authorize_budget_extension import (
    authorize_budget_extension,
)


def configs():
    budgets = {
        "stage_limits": {
            "relax_and_feature": {"max_tasks": 1500, "max_cost": 9000, "task_cost": 1}
        }
    }
    old = {
        "budgets": budgets,
        "qbc": {"budget_limits": deepcopy(budgets)},
        "mlip": {"version": "mace-mh-1"},
    }
    new = deepcopy(old)
    new["budgets"]["stage_limits"]["relax_and_feature"]["max_tasks"] = 10000
    new["qbc"]["budget_limits"]["stage_limits"]["relax_and_feature"]["max_tasks"] = (
        10000
    )
    state = {
        "confirmed_config": old,
        "confirmed_config_version": "old",
        "tasks": [{"status": "completed"}],
    }
    return state, {"config_version": "new", "config": new}


def test_confirmed_task_cap_extension_requires_approval_and_retains_identity():
    state, snapshot = configs()
    original = deepcopy(state)
    assert authorize_budget_extension(state, snapshot)["status"] == "approval_required"
    result = authorize_budget_extension(state, snapshot, user_approved=True)
    assert result["status"] == "extended"
    assert state == original
    assert result["state"]["tasks"] == state["tasks"]
    assert result["state"]["confirmed_config"] == snapshot["config"]
    assert len(result["state"]["config_migrations"][-1]["budget_changes"]) == 2


@pytest.mark.parametrize("field", ["budgets", "mirror", "task_cost", "model"])
def test_decrease_and_scientific_changes_remain_blocked(field):
    state, snapshot = configs()
    if field in {"budgets", "mirror"}:
        budget = (
            snapshot["config"]["budgets"]
            if field == "budgets"
            else snapshot["config"]["qbc"]["budget_limits"]
        )
        budget["stage_limits"]["relax_and_feature"]["max_tasks"] = 1000
        expected = "rejected_budget_decrease"
    elif field == "task_cost":
        snapshot["config"]["qbc"]["budget_limits"]["stage_limits"]["relax_and_feature"][
            "task_cost"
        ] = 2
        expected = "rejected_non_budget_change"
    else:
        snapshot["config"]["mlip"]["version"] = "other"
        expected = "rejected_non_budget_change"
    assert (
        authorize_budget_extension(state, snapshot, user_approved=True)["status"]
        == expected
    )

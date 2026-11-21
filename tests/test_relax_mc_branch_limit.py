from copy import deepcopy
import pytest
from phase_agent.tools.relax_mc_batch.branch_limit import (
    batch_reservation_limits,
    validate_branch_scope,
)
from phase_agent.tools.budget.reserve_budget import reserve_owned_budget


def test_approved_batch_ignores_lifetime_task_counter_but_preserves_other_limits():
    limits = {
        "enforcement": "advisory",
        "stage_limits": {
            "deep_search": {"max_tasks": 300, "task_cost": 1},
            "relax_and_feature": {"max_tasks": 8},
        },
    }
    before = deepcopy(limits)
    scope = [f"b{i}" for i in range(500)]
    scoped = batch_reservation_limits(limits, scope)
    state = {"budget_usage": {"stages": {"deep_search": {"tasks": 610}}}, "budget_reservations": {}}
    assert (
        reserve_owned_budget(state, task_key="new", stage="deep_search", amount=1, limits=scoped)[
            "status"
        ]
        == "reserved"
    )
    assert limits == before and scoped["stage_limits"]["relax_and_feature"]["max_tasks"] == 8
    assert state["budget_usage"]["stages"]["deep_search"]["tasks"] == 610


def test_batch_over_500_or_duplicate_is_rejected():
    with pytest.raises(ValueError, match="branch_limit_exceeded"):
        validate_branch_scope([str(i) for i in range(501)])
    with pytest.raises(ValueError, match="branch_scope_invalid"):
        validate_branch_scope(["same", "same"])

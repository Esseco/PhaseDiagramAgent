import pytest
from decision_layer.agent.mc_contracts import mc_contract_errors
from execution_layer.workflows.create_active_learning_handlers import _allocate_mc_bohb


@pytest.mark.parametrize("params", [{"mc_budget": True}, {"mc_budget": "100"},
    {"mc_budget": 10.5}, {"mc_budget": -1}, {"dft_budget": float("nan")},
    {"exploration_fraction": float("inf")}, {"exploration_fraction": 1.1},
    {"seed": False}, {"focus_regions": [1]}, {"focus_regions": "P3"}])
def test_invalid_fields_rejected_before_science(params):
    assert mc_contract_errors(params)
    state = {"tasks": []}
    result = _allocate_mc_bohb(action={"parameters": params}, context={"event_state": state})
    assert result["status"] == "rejected"
    assert result["state"] == state
    assert state == {"tasks": []}


def test_internal_preview_preserved_and_valid_common_fields():
    params = {"mc_budget": 100, "dft_budget": 0, "seed": 2,
        "exploration_fraction": .2, "focus_regions": ["P3"],
        "budget_preview": {"full_plan_steps": 100}, "round_kind": "second"}
    assert not mc_contract_errors(params)
    assert params["budget_preview"] == {"full_plan_steps": 100}

from phase_agent.configuration.defaults.default_layered_search_config import (
    default_layered_search_config,
)
from phase_agent.science.mc.schedule_tiered_mc import _initial_band, _validate_tiers, _next_tier
import pytest


def test_default_initial_and_followup_budgets():
    policy = default_layered_search_config()["mc_policy"]
    for gap, steps, patience in [(0.005, 100, 10), (0.015, 80, 8), (0.03, 60, 6), (0.05, 40, 4)]:
        actual = _initial_band({"relaxed_ehull": gap}, policy)
        assert actual["max_mc_steps"] == steps
        assert actual["patience_steps"] == patience
    assert [(t["max_mc_steps"], t["patience_steps"]) for t in policy["tiers"]] == [
        (100, 10),
        (200, 20),
        (200, 20),
    ]
    _validate_tiers(policy["tiers"])
    previous = [
        {
            "status": "completed",
            "tier_index": 0,
            "energy_improvement": 0.002,
            "min_improvement": 0.001,
        }
    ]
    index, reason = _next_tier(previous, {"relaxed_ehull": 0.005}, policy["tiers"], policy)
    assert index == 1 and reason == "sustained_improvement_promotion"
    assert policy["tiers"][index]["max_mc_steps"] == 200


@pytest.mark.parametrize("field", ["max_mc_steps", "patience_steps"])
def test_nonpositive_tier_budgets_still_rejected(field):
    tier = {"max_mc_steps": 200, "patience_steps": 20, "min_improvement": 0.001, field: 0}
    with pytest.raises(ValueError, match="positive"):
        _validate_tiers([tier])

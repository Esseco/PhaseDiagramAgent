from phase_agent.analysis.cost.calibrate_relative_cost import calibrate_relative_cost
from phase_agent.configuration.defaults.default_layered_search_config import (
    default_layered_search_config,
)


def test_unknown_costs_do_not_calibrate_or_crash():
    rows = [
        {
            "stage": "deep_search",
            "status": "completed",
            "planned_cost": 10,
            "actual_cost": value,
            "actual_cost_known": False,
        }
        for value in (None, 1)
    ]
    assert calibrate_relative_cost({"cost_history": rows}, "deep_search") == (1.0, 0)


def test_measured_cost_can_calibrate():
    state = {
        "cost_history": [
            {
                "stage": "deep_search",
                "status": "completed",
                "planned_cost": 10,
                "actual_cost": 5,
                "actual_cost_known": True,
            }
        ]
    }
    factor, samples = calibrate_relative_cost(state, "deep_search")
    assert abs(factor - 0.65) < 1e-9 and samples == 1


def test_dft_round_has_task_cap_without_cost_ceiling():
    selection = default_layered_search_config()["dft"]["selection"]
    assert selection["max_tasks_per_round"] == 100
    assert selection["single_point_cost_per_round"] is None

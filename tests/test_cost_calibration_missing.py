from analysis_layer.cost.estimate_proposal_cost import _calibration
from config_layer.defaults.default_layered_search_config import default_layered_search_config


def test_unknown_costs_do_not_calibrate_or_crash():
    rows = [{"stage": "deep_search", "status": "completed", "planned_cost": 10,
             "actual_cost": value, "actual_cost_known": False} for value in (None, 1)]
    assert _calibration({"cost_history": rows}, "deep_search") == (1.0, 0)


def test_measured_cost_can_calibrate():
    state = {"cost_history": [{"stage": "deep_search", "status": "completed",
             "planned_cost": 10, "actual_cost": 5, "actual_cost_known": True}]}
    factor, samples = _calibration(state, "deep_search")
    assert abs(factor - .65) < 1e-9 and samples == 1


def test_single_point_budget_increased():
    assert default_layered_search_config()["dft"]["selection"]["single_point_cost_per_round"] == 5000

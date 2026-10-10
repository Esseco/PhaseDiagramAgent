import pytest
from phase_agent.analysis.cost.estimate_task_cost import estimate_task_cost, estimate_batch_cost


def test_reference_costs_reuse_existing_model():
    assert estimate_task_cost("relax_and_feature", atom_count=40)["scenarios"][0]["estimated_cost"] == 1
    assert estimate_task_cost("dft_single_point", atom_count=40)["scenarios"][0]["estimated_cost"] == 30
    assert estimate_task_cost("dft_relax", atom_count=40)["scenarios"][0]["estimated_cost"] == 900


def test_patience_scenario_is_not_maximum_or_prediction():
    result = estimate_task_cost("deep_search", atom_count=40, patience=20, max_mc_steps=100)
    assert [row["estimated_cost"] for row in result["scenarios"]] == [2, 10]
    assert result["quality"] == "initial_rough_estimate"
    assert result["wall_time"] is None


def test_workload_requires_reference_and_batch_sums():
    with pytest.raises(ValueError):
        estimate_task_cost("dft_relax", relax_steps=50)
    result = estimate_batch_cost([{"stage": "dft_single_point", "atom_count": 40},
        {"stage": "deep_search", "atom_count": 40, "patience": 20, "max_mc_steps": 100}])
    assert result["maximum_scenario_total"] == 40
    assert result["lowest_scenario_total"] == 32


def test_backend_calibration_does_not_mix_samples():
    state = {"cost_history": [{"stage": "relax_and_feature", "status": "completed",
        "actual_cost": 2, "planned_cost": 1, "backend": "other"}]}
    result = estimate_task_cost("relax_and_feature", state=state, backend="mace")
    assert result["measured_samples"] == 0


def test_invalid_patience_is_explicit():
    with pytest.raises(ValueError):
        estimate_task_cost("deep_search", patience=100, max_mc_steps=20)

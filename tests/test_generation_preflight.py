import pytest

from phase_agent.analysis.cost.generation_preflight import preview_generation_cost
from phase_agent.decisions.agent.post_dft_review import valid_post_dft_review
from tests.test_post_dft_review import review, parameters


def framework(size, cost):
    return {
        "phase": "O3",
        "det_H": size,
        "atom_count_upper": 4 * size,
        "allowed_x": [0.5],
        "stage_costs": {
            "relax_and_feature": cost,
            "deep_search": cost,
            "dft_single_point": cost * 3,
        },
        "stage_seconds": {"relax_and_feature": None, "deep_search": None},
    }


def test_preflight_respects_global_cap_and_remaining_budget():
    action = {
        "parameters": {
            "max_det_H": 1,
            "batch_size": 2,
            "generation_plan": [
                {"strategy": "coverage", "phase": "O3", "quota": 2, "max_det_H": 12}
            ],
        }
    }
    result = preview_generation_cost(
        action,
        [framework(1, 10), framework(12, 1000)],
        {"budget_usage": {"total_relative_cost": 20}, "reserved_relative_cost": 5},
        {"budgets": {"total_relative_cost": 60}},
    )
    assert result["relax_mc_cost_upper"] == 40
    assert result["remaining_budget"] == 35
    assert result["status"] == "over_budget"
    assert result["serial_seconds_upper"] is None
    assert result["allocations"][0]["atom_count_upper"] == 4


def test_preflight_rejects_unavailable_mc_steps():
    row = framework(1, 10)
    row["stage_costs"]["deep_search"] = None
    with pytest.raises(ValueError, match="MC步数"):
        preview_generation_cost(
            {
                "parameters": {
                    "batch_size": 1,
                    "generation_plan": [{"strategy": "coverage", "phase": "O3", "quota": 1}],
                }
            },
            [row],
            {},
            {},
        )


def test_finetune_need_cannot_be_replaced_by_search():
    assessment = review()
    assessment["finetune_recommendation"] = "now"
    assert not valid_post_dft_review(
        {"tool": "generate_branches", "parameters": parameters(), "post_dft_review": assessment}
    )
    assessment["choice"] = "revise_strategy"
    assert valid_post_dft_review({"tool": "adjust_strategy", "post_dft_review": assessment})


def test_metadata_estimate_never_builds_supercell(monkeypatch):
    from types import SimpleNamespace
    from pymatgen.core import Structure, Lattice
    from phase_agent.analysis.cost.generation_preflight import generation_framework_costs
    from phase_agent.configuration.defaults.default_budget_rules import default_budget_rules

    def forbidden(*args, **kwargs):
        raise AssertionError("expansion before cost estimate")

    monkeypatch.setattr(Structure, "make_supercell", forbidden)
    parent = Structure(
        Lattice.cubic(4),
        ["Na", "Fe", "O", "O"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
    )
    manager = SimpleNamespace(
        boundary={
            "P": ["O3"],
            "H": [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[2, 0, 0], [0, 1, 0], [0, 0, 1]]],
        }
    )
    rows = generation_framework_costs(
        manager,
        {"O3": parent},
        {"budgets": default_budget_rules(), "mc_policy": {"tiers": [{"max_mc_steps": 10}]}},
        {},
    )
    assert {r["atom_count_upper"] for r in rows} == {4, 8}
    assert all(r["mc_steps_upper"] == 10 for r in rows)


def test_branch_preview_does_not_require_or_include_dft_cost():
    row = framework(1, 10)
    del row["stage_costs"]["dft_single_point"]
    result = preview_generation_cost(
        {
            "parameters": {
                "batch_size": 96,
                "generation_plan": [{"strategy": "coverage", "phase": "O3", "quota": 300}],
            }
        },
        [row],
        {},
        {"budgets": {"total_relative_cost": 50000}},
    )
    assert result["selected_branch_upper"] == 96
    assert result["relax_mc_cost_upper"] == 1920
    assert "dft_cost_scenario_upper" not in result
    assert "dft_single_point_per_branch_upper" not in result["allocations"][0]

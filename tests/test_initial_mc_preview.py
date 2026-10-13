from types import SimpleNamespace
from copy import deepcopy
import pytest
from phase_agent.tools.workflows.prepare_mc_preview import prepare_mc_preview


def test_model_mc_steps_cost_and_coverage_are_distinct(monkeypatch):
    monkeypatch.setattr(
        "phase_agent.analysis.phase.branch_relax_hull.rank_relaxed_branches",
        lambda *args, **kwargs: ([], []),
    )
    preview = {
        "requested_steps": 60,
        "estimated_relative_cost": 12.1,
        "full_plan_steps": 1800,
        "full_plan_branch_count": 30,
        "selected_branch_count": 1,
        "phase_diagram_version": "h",
        "allocation_checksum": "frozen",
        "missing_branch_ids": [],
        "compression_requires_user_choice": True,
    }
    monkeypatch.setattr(
        "phase_agent.tools.workflows.prepare_mc_preview.estimate_branch_mc_budget",
        lambda *a, **kw: preview,
    )
    original = {
        "tool": "allocate_mc_bohb",
        "target_ids": ["B"],
        "decision_source": "llm_agent",
        "budget": 66,
        "parameters": {"mc_budget": 66, "compression_choice": "reduce_branch_count"},
    }
    old = deepcopy(original)
    result = prepare_mc_preview(
        original,
        {"current_branch_hull_version": "v", "branch_hull_batches": {"v": {"version": "v"}}},
        {"manager": SimpleNamespace(data={"branches": {"B": {}}})},
        {},
    )
    assert result["budget"] == 12.1 and result["parameters"]["mc_budget"] == 66
    assert "compression_choice" not in result["parameters"]
    assert result["parameters"]["budget_preview"]["allocation_checksum"] == "frozen"
    assert "1个分支" in result["expected_purpose"] and "1800步" in result["expected_purpose"]
    assert original == old


def test_missing_mc_preview_facts_fail_before_approval():
    with pytest.raises(ValueError, match="台账"):
        prepare_mc_preview({"tool": "allocate_mc_bohb"}, {}, {}, {})

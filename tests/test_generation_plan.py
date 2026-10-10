from types import SimpleNamespace
import pytest
from phase_agent.decisions.agent.generation_plan import validate_generation_plan
from phase_agent.decisions.agent.propose_tool_action import _apply_generation_defaults
from phase_agent.science.structures.propose_branches import propose_branches


def params():
    return {
        "total_quota": 10,
        "batch_size": 5,
        "quotas": {"coverage": 10},
        "generation_plan": [
            {
                "strategy": "coverage",
                "quota": 10,
                "phase": "P3",
                "na_min": 0.2,
                "na_max": 0.6,
                "reason": "补缺失组分",
            }
        ],
    }


def test_explicit_smaller_plan_not_inflated_to_defaults():
    result = _apply_generation_defaults(
        {"parameters": params()},
        {"available_branches": [{}]},
        {"run": {"total_quota": 300, "batch_size": 96}},
    )
    assert result["parameters"]["total_quota"] == 10
    assert result["parameters"]["batch_size"] == 5
    validate_generation_plan(result["parameters"])
    result["parameters"]["quotas"] = {"coverage": 300}
    with pytest.raises(ValueError):
        validate_generation_plan(result["parameters"])


def test_target_is_passed_to_actual_generator():
    h = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    frameworks = [{"P": phase, "H": h, "allowed_x": ["0", "1/2", "1"]} for phase in ("O3", "P3")]
    seen = []

    class Registry:
        def names(self):
            return ["coverage"]

        def get(self, name):
            def generate(context, quota, seed, options):
                seen.extend(context["frameworks"])
                return []

            return generate

    manager = SimpleNamespace(data={"branches": {}}, boundary={})
    propose_branches(
        manager,
        {},
        quotas={"coverage": 10},
        seed=1,
        register=False,
        registry=Registry(),
        framework_enumerator=lambda *a: frameworks,
        strategy_options={"_focus": params()["generation_plan"][0]},
    )
    assert len(seen) == 1 and seen[0]["P"] == "P3" and seen[0]["allowed_x"] == ["1/2"]


def test_enabled_strategy_constraints_reject_single_phase_competing_allocation():
    from phase_agent.decisions.agent.generation_plan import (
        configured_generation_strategies,
        disabled_generation_allocations,
    )
    from phase_agent.decisions.agent.proposal_validation import proposal_errors

    config = {
        "system": {"generation": {"enabled_strategies": ["coverage", "composition"]}},
        "generation_actions": {"enabled": ["coverage", "composition", "competing_phase"]},
    }
    enabled = configured_generation_strategies(config)
    action = {
        "tool": "generate_branches",
        "task_key": "test",
        "target_ids": [],
        "budget": 0,
        "reason": "test",
        "expected_purpose": "test",
        "parameters": {
            "total_quota": 300,
            "quotas": {"coverage": 120, "composition": 96, "competing_phase": 84},
            "generation_plan": [
                {"strategy": k, "quota": v, "phase": "all", "reason": "test"}
                for k, v in {"coverage": 120, "composition": 96, "competing_phase": 84}.items()
            ],
        },
    }
    assert disabled_generation_allocations(action["parameters"], enabled) == ["competing_phase"]
    errors = proposal_errors(
        action,
        {
            "allowed_tools": ["generate_branches"],
            "decision_context": {"enabled_generation_strategies": enabled},
        },
    )
    assert any("disabled strategies" in error for error in errors)
    assert action["parameters"]["quotas"]["competing_phase"] == 84
    assert configured_generation_strategies({"generation_actions": {"enabled": []}}) == []

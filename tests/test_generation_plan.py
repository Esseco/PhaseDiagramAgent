from types import SimpleNamespace
import pytest
from decision_layer.agent.generation_plan import validate_generation_plan
from decision_layer.agent.propose_tool_action import _apply_generation_defaults
from scientific_layer.structures.propose_branches import propose_branches


def params():
    return {"total_quota": 10, "batch_size": 5, "quotas": {"coverage": 10},
            "generation_plan": [{"strategy": "coverage", "quota": 10, "phase": "P3",
                                 "na_min": .2, "na_max": .6, "reason": "补缺失组分"}]}


def test_explicit_smaller_plan_not_inflated_to_defaults():
    result = _apply_generation_defaults({"parameters": params()}, {"available_branches": [{}]},
        {"run": {"total_quota": 300, "batch_size": 96}})
    assert result["parameters"]["total_quota"] == 10
    assert result["parameters"]["batch_size"] == 5
    validate_generation_plan(result["parameters"])
    result["parameters"]["quotas"] = {"coverage": 300}
    with pytest.raises(ValueError):
        validate_generation_plan(result["parameters"])


def test_target_is_passed_to_actual_generator():
    h = [[1,0,0],[0,1,0],[0,0,1]]
    frameworks = [{"P": phase, "H": h, "allowed_x": ["0", "1/2", "1"]} for phase in ("O3", "P3")]
    seen = []
    class Registry:
        def names(self): return ["coverage"]
        def get(self, name):
            def generate(context, quota, seed, options):
                seen.extend(context["frameworks"])
                return []
            return generate
    manager = SimpleNamespace(data={"branches": {}}, boundary={})
    propose_branches(manager, {}, quotas={"coverage": 10}, seed=1, register=False,
        registry=Registry(), framework_enumerator=lambda *a: frameworks,
        strategy_options={"_focus": params()["generation_plan"][0]})
    assert len(seen) == 1 and seen[0]["P"] == "P3" and seen[0]["allowed_x"] == ["1/2"]

import pytest
from pydantic import ValidationError
from decision_layer.agent.generation_contracts import GenerationAllocation
from decision_layer.agent.generation_plan import validate_generation_plan
from tests.test_post_dft_review import parameters


@pytest.mark.parametrize("patch", [{"quota": "10"}, {"quota": True}, {"quota": 1.5},
    {"na_min": True, "na_max": 1}, {"na_min": .2}, {"na_min": .8, "na_max": .2},
    {"na_min": 0, "na_max": float("inf")}, {"max_det_H": "12"}, {"phase": " "}, {"execute": True}])
def test_strict_allocation_rejects_bad_parameters(patch):
    row = {**parameters()["generation_plan"][0], **patch}
    with pytest.raises(ValidationError):
        GenerationAllocation.model_validate(row)


def test_preserves_parameters_and_keeps_quota_consistency():
    params = parameters()
    params["generation_plan"][0].update(na_min=0, na_max=1, max_det_H=12)
    assert validate_generation_plan(params) is params["generation_plan"]
    params["total_quota"] = 11
    with pytest.raises(ValueError, match="total_quota"):
        validate_generation_plan(params)


@pytest.mark.parametrize("patch", [{"total_quota": 10.0}, {"total_quota": True},
                                  {"quotas": {"coverage": "10"}}, {"quotas": []}])
def test_top_level_quota_types_are_not_silently_accepted(patch):
    with pytest.raises(ValueError):
        validate_generation_plan({**parameters(), **patch})

from copy import deepcopy
from phase_agent.science.structures.deduplicate_candidates import deduplicate_candidates
from phase_agent.tools.workflows.region_controller import sample_regions
from phase_agent.agents.exploration.generation_plan import validate_generation_plan
import pytest

H = [[1,0,0],[0,1,0],[0,0,1]]
class Manager:
    def __init__(self):
        self.data = {"branches": {}, "structures": {}}
        self.boundary = {"TM_ratio": {"Fe": 1}}


def review():
    return {
        "tool": "generate_branches",
        "parameters": {"regions": []},
        "branch_pool_review": {
            "choice": "reuse_existing",
            "sufficiency_for_next_search": "按需旧branch续搜",
            "rationale": "依据当前模型",
            "alternative_reason": "新增收益未证实",
            "missing_evidence": [],
            "revisit_conditions": ["回收后复审"],
            "evidence_refs": ["current"],
            "existing_branch_groups": [
                {
                    "decision": "continue_search",
                    "branch_ids": ["a"],
                    "reason": "新模型基态",
                    "evidence_refs": ["current"],
                },
                {
                    "decision": "defer",
                    "branch_ids": ["b"],
                    "reason": "本轮收益不明",
                    "evidence_refs": ["current"],
                },
            ],
        },
    }

def context():
    return {
        "branch_pool_review_required": True,
        "branch_pool_facts": {"existing_branches": [{"branch_id": "a"}, {"branch_id": "b"}]},
        "evidence_catalog": {"references": {"current": {}}},
    }

def test_generated_duplicates_are_left_for_branch_deduplication():
    params = {"sampling_mode":"region_adaptive", "max_new_branches": 1000, "regions":[{"region_id":"r", "priority":1, "sample_count":1000, "strategy":"coverage", "phase":"O3", "min_det_H":1, "max_det_H":1, "na_min":0, "na_max":1, "reason":"test", "evidence_refs":[]}]}
    calls = []
    def producer(*args, **kw):
        count = kw["quotas"]["coverage"]
        calls.append(count)
        return [{"P":"O3", "H":H, "x":"1", "T":["Fe"]} for _ in range(count)]
    manager = Manager()
    chosen, report = sample_regions(manager, {}, params, producer=producer)
    assert sum(calls) == len(chosen) == 1000
    result = deduplicate_candidates(chosen, manager=manager, branch_only=True)
    assert len(result["unique_candidates"]) == 1
    assert result["statistics"]["configuration_duplicates"] == 999
    assert not manager.data["branches"]

def test_existing_branch_is_removed_before_configuration_matching():
    manager = Manager()
    row = {"P":"O3", "H":H, "x":"1", "T":["Fe"]}
    manager.data["branches"]["B-old"] = {**row, "structure_ids": []}
    result = deduplicate_candidates([deepcopy(row)], manager=manager, branch_only=True)
    assert not result["unique_candidates"]
    assert result["duplicate_records"][0]["duplicate_of"] == "B-old"

def test_plan_rejects_more_than_1000_before_generation():
    with pytest.raises(ValueError, match="1000"):
        validate_generation_plan({"generation_plan":[{}], "quotas":{"coverage":1001}, "total_quota":1001})

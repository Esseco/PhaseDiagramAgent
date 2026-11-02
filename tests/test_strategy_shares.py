from copy import deepcopy

from phase_agent.policies.region_sampling import errors
from phase_agent.tools.workflows.region_controller import sample_regions

H = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def region(identity, strategy):
    return {
        "region_id": identity,
        "priority": 1,
        "strategy": strategy,
        "phase": "O3",
        "min_det_H": 1,
        "max_det_H": 2,
        "na_min": 0,
        "na_max": 1,
        "reason": "test",
        "evidence_refs": [],
    }


def parameters():
    return {
        "sampling_mode": "region_adaptive",
        "regions": [region("coverage", "coverage"), region("composition", "composition")],
        "max_new_branches": 100,
    }


def test_strategy_shares_validate_used_strategies_and_sum():
    p = parameters()
    assert "multiple_strategies_require_shares" in errors(p)
    for bad in (
        {"coverage": 0.7},
        {"coverage": 0.7, "composition": 0.2},
        {"coverage": 0.7, "composition": 0.3, "tm_ordering": 0.1},
    ):
        assert any("strategy_shares" in item for item in errors({**p, "strategy_shares": bad}))
    assert not errors({**p, "strategy_shares": {"coverage": 0.7, "composition": 0.3}})


class Manager:
    def __init__(self):
        self.data = {"branches": {}, "structures": {}}

    def add_branch(self, **row):
        self.data["branches"][str(len(self.data["branches"]))] = row


def test_generation_uses_shares_of_accepted_candidates():
    p = parameters()
    p["strategy_shares"] = {"coverage": 0.7, "composition": 0.3}
    manager = Manager()
    before = deepcopy(manager.data)
    counter = 0

    def producer(*args, **kwargs):
        nonlocal counter
        name, work = next(iter(kwargs["quotas"].items()))
        rows = []
        for _ in range(work):
            counter += 1
            rows.append(
                {"P": "O3", "H": H, "x": "1", "T": [str(counter)], "det_H": 1, "strategy": name}
            )
        return rows

    chosen, report = sample_regions(
        manager, {}, p, producer=producer, descriptor=lambda r: r["T"][0]
    )
    counts = report["strategy_accepted_counts"]
    assert len(chosen) == 100
    assert abs(counts["coverage"] - 70) <= 16
    assert sum(counts.values()) == len(chosen)
    assert manager.data == before


def test_parent_driven_region_receives_only_frozen_near_hull_parents():
    p = {
        "sampling_mode": "region_adaptive",
        "regions": [region("x", "composition")],
        "max_new_branches": 1,
        "region_preview": {
            "near_hull_parent_evidence": {"parents": [{"branch_id": "near", "record_id": "r"}]}
        },
    }
    manager = Manager()
    manager.data["branches"] = {
        "near": {"P": "O3", "H": H, "x": "1/2", "T": ["A"]},
        "far": {"P": "O3", "H": H, "x": "1/2", "T": ["B"]},
    }
    calls = []

    def producer(*args, **kwargs):
        calls.append(kwargs["parent_branch_ids"])
        return [
            {
                "P": "O3",
                "H": H,
                "x": "1",
                "T": ["C"],
                "det_H": 1,
                "strategy": "composition",
                "parent_branch_id": "near",
            }
        ]

    chosen, _ = sample_regions(manager, {}, p, producer=producer, descriptor=lambda r: r["T"][0])
    assert calls == [["near"]]
    assert chosen[0]["near_hull_parent_evidence"]["record_id"] == "r"


def test_small_review_still_samples_ten_percent_strategies():
    names = ["coverage", "composition", "competing_phase", "periodic_extension"]
    p = {
        "sampling_mode": "region_adaptive",
        "regions": [region(name, name) for name in names],
        "strategy_shares": {
            "coverage": 0.7,
            "composition": 0.1,
            "competing_phase": 0.1,
            "periodic_extension": 0.1,
        },
        "max_new_branches": 10,
    }
    counter = 0

    def producer(*args, **kwargs):
        nonlocal counter
        strategy, work = next(iter(kwargs["quotas"].items()))
        result = []
        for _ in range(work):
            counter += 1
            result.append(
                {"P": "O3", "H": H, "x": "1", "T": [str(counter)], "det_H": 1, "strategy": strategy}
            )
        return result

    chosen, report = sample_regions(
        Manager(), {}, p, producer=producer, descriptor=lambda r: r["T"][0]
    )
    assert len(chosen) == 10
    assert all(report["strategy_accepted_counts"][name] >= 1 for name in names)

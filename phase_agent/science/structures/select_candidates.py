"""按覆盖、分组上限和随机探索选择候选批次。"""

from __future__ import annotations

import copy
import random
from typing import Any

from phase_agent.science.structures.boundary_utils import (
    compact_json,
    det_H,
    normalize_H,
    normalize_fraction,
)


def select_candidates(
    candidates: list[dict[str, Any]],
    *,
    manager: Any | None = None,
    batch_size: int | None = None,
    cost_budget: float | None = None,
    max_per_framework: int = 8,
    max_per_parent_branch: int = 2,
    random_fraction: float = 0.2,
    seed: int = 0,
) -> dict[str, Any]:
    """选择批次；可同时受数量和相对成本预算约束。"""
    if batch_size is None and cost_budget is None:
        raise ValueError("batch_size 和 cost_budget 至少提供一个")
    if batch_size is not None:
        _positive_int(batch_size, "batch_size")
    _positive_int(max_per_framework, "max_per_framework")
    _positive_int(max_per_parent_branch, "max_per_parent_branch")
    if not isinstance(random_fraction, (int, float)) or not 0 <= random_fraction <= 1:
        raise ValueError("random_fraction 必须在 [0,1]")
    if cost_budget is not None and (
        isinstance(cost_budget, bool)
        or not isinstance(cost_budget, (int, float))
        or cost_budget < 0
    ):
        raise ValueError("cost_budget 必须是非负数")

    pool = [copy.copy(item) for item in candidates]
    pool = [
        item
        for item in pool
        if not (item.get("deduplication") or {}).get("configuration_duplicate", False)
    ]
    coverage = _historical_coverage(manager)
    phase_counts = _historical_phase_coverage(manager)
    phase_x_counts = _historical_phase_x_coverage(manager)
    rng = random.Random(seed)
    rng.shuffle(pool)
    random_count = round((batch_size or len(pool)) * random_fraction)
    exploration = pool[:random_count]
    exploration_ids = {id(item) for item in exploration}
    remaining = pool[random_count:]

    selected, rejected = [], []
    framework_counts: dict[str, int] = {}
    parent_counts: dict[str, int] = {}
    total_cost = 0.0
    while exploration or remaining:
        if exploration:
            candidate = exploration.pop(0)
        else:
            index = min(
                range(len(remaining)),
                key=lambda index: _selection_key(
                    remaining[index],
                    phase_counts,
                    phase_x_counts,
                    coverage,
                    cost_budget is not None,
                ),
            )
            candidate = remaining.pop(index)
        framework = _framework_key(candidate)
        parent = candidate.get("parent_branch_id")
        cost = _cost(candidate, cost_budget is not None)
        reason = None
        if framework_counts.get(framework, 0) >= max_per_framework:
            reason = "framework_limit"
        elif parent is not None and parent_counts.get(parent, 0) >= max_per_parent_branch:
            reason = "parent_branch_limit"
        elif batch_size is not None and len(selected) >= batch_size:
            reason = "batch_size"
        elif cost_budget is not None and total_cost + cost > cost_budget + 1e-12:
            reason = "cost_budget"
        if reason:
            rejected.append({"candidate": candidate, "reason": reason})
            continue
        candidate["selection"] = {
            "mode": "random_exploration" if id(candidate) in exploration_ids else "coverage",
            "previous_region_count": coverage.get(_region_key(candidate), 0),
        }
        selected.append(candidate)
        total_cost += cost
        framework_counts[framework] = framework_counts.get(framework, 0) + 1
        if parent is not None:
            parent_counts[parent] = parent_counts.get(parent, 0) + 1
        coverage[_region_key(candidate)] = coverage.get(_region_key(candidate), 0) + 1
        phase_counts[candidate["P"]] = phase_counts.get(candidate["P"], 0) + 1
        phase_x = (candidate["P"], normalize_fraction(candidate["x"]))
        phase_x_counts[phase_x] = phase_x_counts.get(phase_x, 0) + 1

    return {
        "selected_candidates": selected,
        "rejected_candidates": rejected,
        "summary": {
            "input": len(candidates),
            "eligible": len(pool),
            "selected": len(selected),
            "selected_relative_cost": total_cost,
            "batch_size": batch_size,
            "cost_budget": cost_budget,
            "seed": seed,
        },
    }


def _historical_coverage(manager):
    counts: dict[str, int] = {}
    if manager is None:
        return counts
    for branch in manager.data.get("branches", {}).values():
        key = _region_key(branch)
        counts[key] = counts.get(key, 0) + len(branch.get("structure_ids", []))
    return counts


def _historical_phase_coverage(manager):
    counts = {}
    for branch in manager.data.get("branches", {}).values() if manager else []:
        counts[branch["P"]] = counts.get(branch["P"], 0) + 1
    return counts


def _historical_phase_x_coverage(manager):
    counts = {}
    for branch in manager.data.get("branches", {}).values() if manager else []:
        key = (branch["P"], normalize_fraction(branch["x"]))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _selection_key(item, phase_counts, phase_x_counts, coverage, cost_required):
    return (
        phase_counts.get(item["P"], 0),
        phase_x_counts.get((item["P"], normalize_fraction(item["x"])), 0),
        coverage.get(_region_key(item), 0),
        _cost(item, cost_required),
        det_H(item["H"]),
        compact_json([item.get("P"), item.get("H"), item.get("x"), item.get("T")]),
    )


def _region_key(item):
    return compact_json(
        [str(item["P"]).upper(), normalize_H(item["H"]), normalize_fraction(item["x"])]
    )


def _framework_key(item):
    return compact_json([str(item["P"]).upper(), normalize_H(item["H"])])


def _cost(candidate, required):
    record = candidate.get("estimated_cost")
    if record is None:
        if required:
            raise ValueError("使用 cost_budget 时每个候选必须先估计成本")
        return 0.0
    value = record.get("value") if isinstance(record, dict) else record
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError("estimated_cost.value 必须是非负数")
    return float(value)


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} 必须是正整数")

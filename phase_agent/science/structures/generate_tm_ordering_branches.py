"""重排固定组成的 TM 位点，生成新的有序构型。"""

from __future__ import annotations

from collections import Counter
import random
from typing import Any

from phase_agent.science.structures.generate_branch_structure import generate_branch_structure
from phase_agent.science.structures.identify_branch import identify_branch_parameters


def generate_tm_ordering_branches(
    manager: Any,
    phase_references: dict[str, Any],
    *,
    parent_branch_ids: list[str],
    quota: int,
    seed: int,
    oxidation_states: dict[str, int | float] | None = None,
) -> list[dict[str, Any]]:
    """交换异种 TM 的位点；元素种类、计数和总体比例均不可改变。"""
    rng = random.Random(seed)
    tasks = []
    for parent_id in parent_branch_ids:
        parent = _branch(manager, parent_id)
        parent_T = list(parent["T"])
        _validate_fixed_tm_composition(parent_T, manager.boundary["TM_ratio"])
        pairs = [
            (left, right)
            for left in range(len(parent_T))
            for right in range(left + 1, len(parent_T))
            if parent_T[left] != parent_T[right]
        ]
        rng.shuffle(pairs)
        tasks.extend((parent_id, parent, pair) for pair in pairs)
    rng.shuffle(tasks)

    candidates, seen = [], set()
    for offset, (parent_id, parent, (left, right)) in enumerate(tasks):
        parent_T = list(parent["T"])
        T = list(parent_T)
        T[left], T[right] = T[right], T[left]
        _assert_same_tm_composition(parent_T, T)
        key = (parent_id, tuple(T))
        if key in seen:
            continue
        seen.add(key)
        current_seed = seed + offset
        structure = generate_branch_structure(
            manager.boundary,
            phase=parent["P"],
            H=parent["H"],
            x=parent["x"],
            T=T,
            phase_references=phase_references,
            oxidation_states=oxidation_states,
            seed=current_seed,
        )
        parameters = identify_branch_parameters(
            structure,
            manager.boundary,
            phase_references=phase_references,
            phase_hint=parent["P"],
        )
        _assert_same_tm_composition(parent_T, parameters["T"])
        candidates.append(
            {
                **{name: parameters[name] for name in ("P", "H", "det_H", "x", "T", "composition")},
                "structure": structure,
                "strategy": "tm_ordering",
                "parent_branch_id": parent_id,
                "seed": current_seed,
                "inheritance": "TM_site_swap",
                "changes": {"swapped_sites": [left, right]},
            }
        )
        if len(candidates) >= quota:
            break
    return candidates


def _validate_fixed_tm_composition(T: list[str], ratio: dict[str, int | float]) -> None:
    expected_elements = set(ratio)
    actual = Counter(T)
    if set(actual) != expected_elements:
        raise ValueError(
            f"父 branch 的 TM 元素 {sorted(actual)} 与冻结边界 {sorted(expected_elements)} 不一致"
        )
    scales = [actual[element] / amount for element, amount in ratio.items()]
    if any(abs(value - scales[0]) > 1e-8 for value in scales):
        raise ValueError(f"父 branch 的 TM 计数 {dict(actual)} 不满足冻结比例 {ratio}")


def _assert_same_tm_composition(parent_T: list[str], candidate_T: list[str]) -> None:
    if Counter(parent_T) != Counter(candidate_T):
        raise RuntimeError("TM 排布策略改变了冻结组成，已拒绝该候选")


def _branch(manager: Any, branch_id: str) -> dict[str, Any]:
    try:
        return manager.data["branches"][branch_id]
    except KeyError as error:
        raise KeyError(f"未知父 branch：{branch_id}") from error

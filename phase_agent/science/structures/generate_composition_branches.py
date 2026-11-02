"""保持 P、H、T，仅改变可实现的 x。"""

from __future__ import annotations

import random
from fractions import Fraction
from typing import Any

from phase_agent.science.structures.identify_branch import identify_branch_parameters
from phase_agent.science.structures.generate_branch_structure import generate_branch_structure


def generate_composition_branches(
    manager: Any,
    frameworks: list[dict[str, Any]],
    phase_references: dict[str, Any],
    *,
    parent_branch_ids: list[str],
    quota: int,
    seed: int,
    oxidation_states: dict[str, int | float] | None = None,
    preserve_reference_tm: bool = False,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    tasks = []
    # Focused regions may omit nearer compositions; use the full legal grid
    # to identify neighbours before intersecting with the approved region.
    from phase_agent.science.structures.enumerate_legal_frameworks import enumerate_legal_frameworks

    legal = enumerate_legal_frameworks(manager.boundary, phase_references)
    for parent_id in parent_branch_ids:
        parent = _branch(manager, parent_id)
        framework = _framework(frameworks, parent["P"], parent["H"])
        full = _framework(legal, parent["P"], parent["H"])
        neighbours = _neighbouring_x(parent["x"], full["allowed_x"])
        for x in framework["allowed_x"]:
            if Fraction(str(x)) in neighbours:
                tasks.append((parent_id, parent, x))
    rng.shuffle(tasks)

    candidates = []
    for offset, (parent_id, parent, x) in enumerate(tasks[:quota]):
        current_seed = seed + offset
        structure = generate_branch_structure(
            manager.boundary,
            phase=parent["P"],
            H=parent["H"],
            x=x,
            T=None if preserve_reference_tm else parent["T"],
            preserve_reference_tm=preserve_reference_tm,
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
        candidates.append(_candidate(parameters, structure, "composition", parent_id, current_seed))
    return candidates


def _neighbouring_x(parent_x, allowed_x):
    """Nearest legal composition on each side, independent of region filtering."""
    parent = Fraction(str(parent_x))
    ordered = sorted({Fraction(str(value)) for value in allowed_x})
    below = [value for value in ordered if value < parent]
    above = [value for value in ordered if value > parent]
    return set(below[-1:] + above[:1])


def _branch(manager: Any, branch_id: str) -> dict[str, Any]:
    try:
        return manager.data["branches"][branch_id]
    except KeyError as error:
        raise KeyError(f"未知父 branch：{branch_id}") from error


def _framework(frameworks: list[dict[str, Any]], P: str, H: Any) -> dict[str, Any]:
    matches = [item for item in frameworks if item["P"] == P and item["H"] == H]
    if len(matches) != 1:
        raise ValueError(f"无法为 P={P}, H={H} 唯一确定框架")
    return matches[0]


def _candidate(
    parameters: dict[str, Any], structure: Any, strategy: str, parent: str, seed: int
) -> dict[str, Any]:
    return {
        **{key: parameters[key] for key in ("P", "H", "det_H", "x", "T", "composition")},
        "structure": structure,
        "strategy": strategy,
        "parent_branch_id": parent,
        "seed": seed,
        "inheritance": "preserved_T",
    }

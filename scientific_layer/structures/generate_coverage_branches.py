"""向现有候选较少的合法 P/H/x 区域补充 branch。"""

from __future__ import annotations

from typing import Any

from scientific_layer.structures.identify_branch import identify_branch_parameters
from scientific_layer.structures.generate_branch_structure import generate_branch_structure


def generate_coverage_branches(
    manager: Any,
    frameworks: list[dict[str, Any]],
    phase_references: dict[str, Any],
    *,
    quota: int,
    seed: int,
    oxidation_states: dict[str, int | float] | None = None,
) -> list[dict[str, Any]]:
    """按当前 P/H/x branch 数升序生成候选。"""
    regions = []
    for framework in frameworks:
        for x in framework["allowed_x"]:
            count = sum(
                branch["P"] == framework["P"]
                and branch["H"] == framework["H"]
                and branch["x"] == x
                for branch in manager.data["branches"].values()
            )
            regions.append((count, framework["P"], str(framework["H"]), x, framework))
    regions.sort(key=lambda item: item[:4])

    candidates = []
    for offset, (_, _, _, x, framework) in enumerate(regions[:quota]):
        current_seed = seed + offset
        structure = generate_branch_structure(
            manager.boundary,
            phase=framework["P"],
            H=framework["H"],
            x=x,
            phase_references=phase_references,
            oxidation_states=oxidation_states,
            seed=current_seed,
        )
        parameters = identify_branch_parameters(
            structure, manager.boundary, phase_references=phase_references
        )
        candidates.append(
            _candidate(
                parameters, structure, "coverage", None, current_seed, "generated"
            )
        )
    return candidates


def _candidate(
    parameters: dict[str, Any],
    structure: Any,
    strategy: str,
    parent: str | None,
    seed: int,
    inheritance: str,
) -> dict[str, Any]:
    return {
        **{
            key: parameters[key] for key in ("P", "H", "det_H", "x", "T", "composition")
        },
        "structure": structure,
        "strategy": strategy,
        "parent_branch_id": parent,
        "seed": seed,
        "inheritance": inheritance,
    }

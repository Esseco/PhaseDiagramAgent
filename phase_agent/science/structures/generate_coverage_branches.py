"""向现有候选较少的合法 P/H/x 区域补充 branch。"""

from __future__ import annotations

from typing import Any

from phase_agent.science.structures.boundary_utils import det_H
from phase_agent.science.structures.identify_branch import identify_branch_parameters
from phase_agent.science.structures.generate_branch_structure import generate_branch_structure


def generate_coverage_branches(
    manager: Any,
    frameworks: list[dict[str, Any]],
    phase_references: dict[str, Any],
    *,
    quota: int,
    seed: int,
    oxidation_states: dict[str, int | float] | None = None,
    preserve_reference_tm: bool = False,
) -> list[dict[str, Any]]:
    """按当前 P/H/x branch 数升序生成候选。"""
    regions_by_phase = {}
    for framework in frameworks:
        for x in framework["allowed_x"]:
            count = sum(
                branch["P"] == framework["P"] and branch["H"] == framework["H"] and branch["x"] == x
                for branch in manager.data["branches"].values()
            )
            regions_by_phase.setdefault(framework["P"], []).append(
                (
                    count,
                    framework.get("det_H", det_H(framework["H"])),
                    framework.get("atom_count_full", 0),
                    x,
                    str(framework["H"]),
                    framework,
                )
            )
    for regions in regions_by_phase.values():
        regions.sort(key=lambda item: item[:5])

    # Cover every allowed phase early; within a phase, prefer smaller cells.
    candidates = []
    offset = 0
    while len(candidates) < quota and any(regions_by_phase.values()):
        # Keep covering sparse regions, then revisit them with a new TM seed.
        # A fixed-T system has no additional branch at the same P/H/x.
        phase = min(
            (name for name, rows in regions_by_phase.items() if rows),
            key=lambda name: (
                regions_by_phase[name][0][0],
                sum(item[0] for item in regions_by_phase[name]),
                name,
            ),
        )
        count, size, atoms, x, matrix_key, framework = regions_by_phase[phase].pop(0)
        current_seed = seed + offset
        structure = generate_branch_structure(
            manager.boundary,
            phase=framework["P"],
            H=framework["H"],
            x=x,
            phase_references=phase_references,
            oxidation_states=oxidation_states,
            preserve_reference_tm=preserve_reference_tm,
            seed=current_seed,
        )
        parameters = identify_branch_parameters(
            structure,
            manager.boundary,
            phase_references=phase_references,
            phase_hint=framework["P"],
        )
        candidates.append(
            _candidate(parameters, structure, "coverage", None, current_seed, "generated")
        )
        if not preserve_reference_tm:
            regions_by_phase[phase].append((count + 1, size, atoms, x, matrix_key, framework))
            regions_by_phase[phase].sort(key=lambda item: item[:5])
        offset += 1
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
        **{key: parameters[key] for key in ("P", "H", "det_H", "x", "T", "composition")},
        "structure": structure,
        "strategy": strategy,
        "parent_branch_id": parent,
        "seed": seed,
        "inheritance": inheritance,
    }

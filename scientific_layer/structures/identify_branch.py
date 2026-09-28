"""从结构识别 branch 参数 P、H、det_H、x、T。"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import (
    SimplestChemenvStrategy,
)
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import (
    LocalGeometryFinder,
)
from pymatgen.analysis.chemenv.utils.defs_utils import AdditionalConditions
from pymatgen.core import Element, Structure

from scientific_layer.structures.boundary_utils import (
    allowed_phases,
    allowed_phases_at_x,
    allowed_supercells,
    det_H,
    load_structure,
    normalize_fraction,
)


def identify_branch_parameters(
    structure: Structure | str | Path | Any,
    boundary: dict[str, Any],
    *,
    phase_references: dict[str, Structure | str | Path],
    lattice_tolerance: float = 0.25,
    ambiguity_tolerance: float = 1e-6,
    layer_tolerance: float = 0.08,
    use_coordination_phase: bool = True,
    phase_hint: str | None = None,
) -> dict[str, Any]:
    present = load_structure(structure)
    references = {
        str(phase).upper(): load_structure(reference)
        for phase, reference in phase_references.items()
    }
    if not references:
        raise ValueError("phase_references 不能为空")

    diagnostic = {"phase": None, "method": "coordination_disabled"}
    if phase_hint is not None:
        phase_hint = str(phase_hint).upper()
        if phase_hint not in references:
            raise ValueError(f"phase_hint {phase_hint!r} 没有对应母结构")
        diagnostic = {"phase": phase_hint, "method": "explicit_generation_label"}
    elif use_coordination_phase:
        try:
            diagnostic = identify_phase(present, layer_tolerance=layer_tolerance)
            phase_hint = None if diagnostic["phase"] == "X" else diagnostic["phase"]
        except (ValueError, RuntimeError, IndexError) as error:
            diagnostic = {
                "phase": None,
                "method": "coordination_failed",
                "error": str(error),
            }

    phase, matrix, error = identify_supercell(
        present,
        references,
        boundary,
        phase_hint=phase_hint,
        tolerance=lattice_tolerance,
        ambiguity_tolerance=ambiguity_tolerance,
    )
    x = calculate_x(present)
    if phase not in allowed_phases_at_x(boundary["P"], x):
        raise ValueError(f"识别出相 {phase!r}、x={x}，但不符合 boundary.P 的组分规则")
    T, site_order = extract_T(present, boundary["TM_ratio"])
    composition = {
        element: _clean_number(amount)
        for element, amount in sorted(present.composition.get_el_amt_dict().items())
    }
    return {
        "P": phase,
        "H": matrix,
        "det_H": det_H(matrix),
        "x": x,
        "T": T,
        "composition": composition,
        "tm_site_order": site_order,
        "identification": {
            "phase": diagnostic,
            "lattice_match_error": error,
            "lattice_tolerance": lattice_tolerance,
        },
    }


def identify_supercell(
    structure: Structure,
    references: dict[str, Structure],
    boundary: dict[str, Any],
    *,
    phase_hint: str | None,
    tolerance: float,
    ambiguity_tolerance: float,
) -> tuple[str, list[list[int]], float]:
    phases = allowed_phases(boundary["P"])
    missing = sorted(phases - set(references))
    if missing:
        raise ValueError(f"缺少允许相的母结构：{missing}")
    unknown = sorted(set(references) - phases)
    if unknown:
        raise ValueError(f"phase_references 含边界外相：{unknown}")

    candidates = []
    current = np.asarray(structure.lattice.matrix, dtype=float)
    for phase in [phase_hint] if phase_hint in references else sorted(references):
        reference = np.asarray(references[phase].lattice.matrix, dtype=float)
        inferred = np.linalg.solve(reference.T, current.T).T
        for matrix in allowed_supercells(boundary["H"], phase):
            error = float(np.max(np.abs(inferred[:2, :2] - np.asarray(matrix)[:2, :2])))
            candidates.append((error, phase, matrix))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    if not candidates or candidates[0][0] > tolerance:
        error = None if not candidates else candidates[0][0]
        raise ValueError(f"没有匹配的 P/H；最小层内晶格误差={error}")
    best = candidates[0]
    ambiguous = [
        item
        for item in candidates[1:]
        if abs(item[0] - best[0]) <= ambiguity_tolerance
        and (item[1], item[2]) != (best[1], best[2])
    ]
    if ambiguous:
        raise ValueError(f"P/H 匹配不唯一：{[best, *ambiguous]}")
    return best[1], best[2], best[0]


def calculate_x(structure: Structure, tolerance: float = 1e-8) -> str:
    composition = structure.composition.get_el_amt_dict()
    oxygen = float(composition.get("O", 0))
    if oxygen <= 0:
        raise ValueError("结构中没有 O")
    value = 2 * float(composition.get("Na", 0)) / oxygen
    if not -tolerance <= value <= 1 + tolerance:
        raise ValueError(f"x={value:.8g} 不在 [0,1]")
    fraction = Fraction(value).limit_denominator(10_000)
    if abs(float(fraction) - value) > tolerance:
        raise ValueError(f"x={value:.12g} 无法可靠有理化")
    return normalize_fraction(fraction)


def extract_T(
    structure: Structure, ratio: dict[str, int | float]
) -> tuple[list[str], list[dict[str, Any]]]:
    elements = set(ratio)
    sites = []
    actual_tm = set()
    for site in structure:
        if not site.is_ordered:
            if set(site.species.as_dict()) & elements:
                raise ValueError("TM 位点含部分占位")
            continue
        symbol = site.specie.symbol
        if Element(symbol).is_transition_metal:
            actual_tm.add(symbol)
        if symbol in elements:
            coords = np.mod(np.asarray(site.frac_coords, dtype=float), 1.0)
            coords[np.isclose(coords, 1, atol=1e-8)] = 0
            sites.append((tuple(round(float(item), 8) for item in coords), symbol))
    if not sites:
        raise ValueError(f"未找到 TM 元素：{sorted(elements)}")
    if actual_tm - elements:
        raise ValueError(f"结构含边界外 TM 元素：{sorted(actual_tm - elements)}")
    sites.sort(key=lambda item: (item[0][2], item[0][1], item[0][0], item[1]))
    order = [symbol for _, symbol in sites]
    counts = Counter(order)
    scales = [counts[element] / value for element, value in ratio.items()]
    if any(abs(value - scales[0]) > 1e-8 for value in scales):
        raise ValueError(f"TM 计数 {dict(counts)} 不满足比例 {ratio}")
    details = [
        {"index": index, "frac_coords": list(coords), "element": symbol}
        for index, (coords, symbol) in enumerate(sites)
    ]
    return order, details


def identify_phase(
    structure: Structure, *, layer_tolerance: float = 0.08, cation: str = "Na"
) -> dict[str, Any]:
    indices = [
        index
        for index, site in enumerate(structure)
        if site.is_ordered and site.specie.symbol == cation
    ]
    if not indices:
        raise ValueError(f"结构中没有 {cation}，需用母结构识别相")
    strategy = SimplestChemenvStrategy(
        additional_condition=AdditionalConditions.ONLY_ELEMENT_TO_OXYGEN_BONDS,
        continuous_symmetry_measure_cutoff=20,
    )
    environments = LocalGeometryFinder().compute_coordination_environments(
        structure,
        indices=indices,
        only_cations=False,
        strategy=strategy,
        valences="undefined",
    )
    labels = _cluster_layers(
        [float(structure[i].frac_coords[2]) for i in indices], layer_tolerance
    )
    site_types = {}
    for index in indices:
        environment = (
            environments.get(index, [])
            if isinstance(environments, dict)
            else environments[index]
        )
        symbol = environment[0].get("ce_symbol") if environment else None
        site_types[index] = {"O:6": "O", "T:6": "P"}.get(symbol, "X")
    layers = []
    for label in range(max(labels) + 1):
        values = [
            site_types[index]
            for index, current in zip(indices, labels, strict=True)
            if current == label and site_types[index] != "X"
        ]
        layers.append(Counter(values).most_common(1)[0][0] if values else "X")
    types = set(layers)
    phase = (
        f"O{len(layers)}"
        if types == {"O"}
        else f"P{len(layers)}"
        if types == {"P"}
        else f"OP{len(layers)}"
        if types == {"O", "P"}
        else "X"
    )
    return {
        "phase": phase,
        "method": "Na_coordination_layers",
        "cation_layer_types": layers,
    }


def _cluster_layers(values: list[float], tolerance: float) -> list[int]:
    wrapped = np.mod(np.asarray(values), 1)
    order = np.argsort(wrapped)
    sorted_values = wrapped[order]
    gaps = np.diff(np.r_[sorted_values, sorted_values[0] + 1])
    start = (int(np.argmax(gaps)) + 1) % len(values)
    rotated_order = np.r_[order[start:], order[:start]]
    rotated = np.r_[sorted_values[start:], sorted_values[:start] + 1]
    labels = np.empty(len(values), dtype=int)
    label, previous = 0, rotated[0]
    for position, value in zip(rotated_order, rotated, strict=True):
        if value - previous > tolerance:
            label += 1
        labels[position], previous = label, value
    return labels.tolist()


def _clean_number(value: float) -> int | float:
    rounded = round(float(value))
    return rounded if abs(float(value) - rounded) <= 1e-10 else float(value)

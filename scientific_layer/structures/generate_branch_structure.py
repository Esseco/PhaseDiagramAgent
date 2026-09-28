"""由 P、H、x 和 TM 比例生成一个具体 branch 结构。"""

from __future__ import annotations

import random
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.core import Element, Structure

from scientific_layer.structures.boundary_utils import (
    allowed_phases,
    allowed_phases_at_x,
    allowed_supercells,
    load_structure,
    normalize_H,
    normalize_fraction,
)


def generate_branch_structure(
    boundary: dict[str, Any],
    *,
    phase: str,
    H: Any,
    x: int | float | str | Fraction,
    phase_references: dict[str, Structure | str | Path],
    element_ratio: dict[str, int | float] | None = None,
    T: list[str] | None = None,
    preserve_reference_tm: bool = False,
    oxidation_states: dict[str, int | float] | None = None,
    seed: int | None = None,
    enforce_phase_composition: bool = True,
) -> Structure:
    phase = phase.upper()
    if phase not in allowed_phases(boundary["P"]):
        raise ValueError(f"相 {phase!r} 不在 boundary['P'] 中")
    if enforce_phase_composition and phase not in allowed_phases_at_x(boundary["P"], x):
        raise ValueError(f"相 {phase!r} 不允许在 x={x} 生成")
    references = {str(key).upper(): value for key, value in phase_references.items()}
    if phase not in references:
        raise ValueError(f"缺少相 {phase!r} 的母结构")
    matrix = normalize_H(H)
    if matrix not in allowed_supercells(boundary["H"], phase):
        raise ValueError(f"H={matrix!r} 不在相 {phase!r} 的允许范围")

    structure = load_structure(references[phase])
    structure.make_supercell(matrix)
    if preserve_reference_tm:
        if T is not None:
            raise ValueError("固定 T 时不能同时传入重新排列的 T")
        from scientific_layer.structures.identify_branch import extract_T
        extract_T(structure, element_ratio or boundary["TM_ratio"])
    else:
        _assign_tm(structure, element_ratio or boundary["TM_ratio"], seed, T)
    structure = _assign_na(structure, x, oxidation_states)
    structure.sort()
    return structure


def _assign_tm(
    structure: Structure,
    ratio: dict[str, int | float],
    seed: int | None,
    requested_T: list[str] | None,
) -> None:
    if not ratio or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0
        for value in ratio.values()
    ):
        raise ValueError("element_ratio 必须是非空正数 dict")
    indices = [
        index
        for index, site in enumerate(structure)
        if site.is_ordered and Element(site.specie.symbol).is_transition_metal
    ]
    total = sum(ratio.values())
    counts = {element: len(indices) * value / total for element, value in ratio.items()}
    if any(abs(value - round(value)) > 1e-8 for value in counts.values()):
        raise ValueError(f"{len(indices)} 个 TM 位点无法满足比例 {ratio}")
    if requested_T is None:
        elements = [
            element for element, count in counts.items() for _ in range(round(count))
        ]
        random.Random(seed).shuffle(elements)
    else:
        elements = list(requested_T)
        expected = {element: round(count) for element, count in counts.items()}
        actual = {element: elements.count(element) for element in ratio}
        if len(elements) != len(indices) or actual != expected:
            raise ValueError(f"T 不满足位点数或比例：{actual}，期望 {expected}")
    ordered_indices = sorted(
        indices,
        key=lambda index: _tm_position_key(structure[index].frac_coords),
    )
    for index, element in zip(ordered_indices, elements, strict=True):
        structure.replace(index, element)


def _tm_position_key(frac_coords: Any) -> tuple[float, float, float]:
    coords = np.mod(np.asarray(frac_coords, dtype=float), 1.0)
    coords[np.isclose(coords, 1.0, atol=1e-8)] = 0.0
    x, y, z = (round(float(value), 8) for value in coords)
    return z, y, x


def _assign_na(
    structure: Structure,
    x: int | float | str | Fraction,
    oxidation_states: dict[str, int | float] | None,
) -> Structure:
    indices = [
        index
        for index, site in enumerate(structure)
        if site.is_ordered and site.specie.symbol == "Na"
    ]
    oxygen = sum(site.is_ordered and site.specie.symbol == "O" for site in structure)
    fraction = Fraction(normalize_fraction(x))
    if not 0 <= fraction <= 1:
        raise ValueError("x 必须在 0～1 之间")
    target = fraction * oxygen / 2
    if target.denominator != 1:
        raise ValueError(f"x={fraction} 对应 {target} 个 Na，无法形成整数占位")
    count = target.numerator
    if count > len(indices):
        raise ValueError(f"需要 {count} 个 Na，但只有 {len(indices)} 个 Na 位点")
    if count == 0:
        structure.remove_species(["Na"])
        return structure
    if count == len(indices):
        return structure

    occupancy = count / len(indices)
    for index in indices:
        structure.replace(index, {"Na": occupancy})
    if oxidation_states is None:
        try:
            structure.add_oxidation_state_by_guess()
        except ValueError as error:
            _assign_charge_balanced_average_tm_valence(structure, fraction, error)
    else:
        structure.add_oxidation_state_by_element(oxidation_states)
    from Process_Vasp.structure import gen_ESGS_structure

    generated = gen_ESGS_structure(structure, 1)
    if not generated:
        raise RuntimeError("gen_ESGS_structure 未生成 Na/空位有序结构")
    ordered = generated[0]
    ordered.remove_oxidation_states()
    return ordered


def _assign_charge_balanced_average_tm_valence(structure, x, cause):
    """Fallback for fractional occupancy: neutral average TM valence for Ewald ranking."""
    composition = structure.composition.get_el_amt_dict()
    oxygen_count = float(composition.get("O", 0))
    alkali = {symbol for symbol in composition if symbol in {"Li", "Na", "K", "Rb", "Cs"}}
    alkali_count = sum(float(composition[symbol]) for symbol in alkali)
    tm_symbols = sorted(
        symbol for symbol in composition
        if Element(symbol).is_transition_metal and symbol not in alkali
    )
    tm_count = sum(float(composition[symbol]) for symbol in tm_symbols)
    other = sorted(set(composition) - set(tm_symbols) - alkali - {"O"})
    if oxygen_count <= 0 or tm_count <= 0 or other:
        raise ValueError(
            f"x={x} 无法自动生成电中性静电排序电荷；未知元素={other}。"
            "请在生成配置中显式提供 oxidation_states。"
        ) from cause
    average_tm_charge = (2.0 * oxygen_count - alkali_count) / tm_count
    states = {symbol: 1.0 for symbol in alkali}
    states["O"] = -2.0
    states.update({symbol: average_tm_charge for symbol in tm_symbols})
    try:
        structure.add_oxidation_state_by_element(states)
    except (ValueError, KeyError) as error:
        raise ValueError(
            f"x={x} 按电中性计算平均 TM 氧化态失败；请显式提供 oxidation_states。"
        ) from error

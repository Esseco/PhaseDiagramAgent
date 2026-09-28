"""列出 boundary 中合法的框架 F=(P,H) 和离散 x。"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from pymatgen.core import Element

from scientific_layer.structures.boundary_utils import (
    allowed_phases,
    allowed_phases_at_x,
    allowed_supercells,
    det_H,
    load_structure,
    normalize_fraction,
    validate_boundary,
)


def enumerate_legal_frameworks(
    boundary: dict[str, Any], phase_references: dict[str, Any]
) -> list[dict[str, Any]]:
    """返回 P、H、det_H、可实现 x 及位点计数。"""
    boundary = validate_boundary(boundary)
    references = {str(key).upper(): value for key, value in phase_references.items()}
    phases = allowed_phases(boundary["P"])
    missing = sorted(phases - set(references))
    if missing:
        raise ValueError(f"缺少允许相的母结构：{missing}")

    frameworks = []
    for phase in sorted(phases):
        for H in allowed_supercells(boundary["H"], phase):
            structure = load_structure(references[phase])
            structure.make_supercell(H)
            oxygen = sum(
                site.is_ordered and site.specie.symbol == "O" for site in structure
            )
            sodium = sum(
                site.is_ordered and site.specie.symbol == "Na" for site in structure
            )
            tm_count = sum(
                site.is_ordered and Element(site.specie.symbol).is_transition_metal
                for site in structure
            )
            _check_tm_ratio(tm_count, boundary["TM_ratio"])
            x_values = {
                Fraction(2 * count, oxygen)
                for count in range(sodium + 1)
                if oxygen and Fraction(2 * count, oxygen) <= 1
                and phase in allowed_phases_at_x(boundary["P"], Fraction(2 * count, oxygen))
            }
            if not x_values:
                continue
            frameworks.append(
                {
                    "P": phase,
                    "H": H,
                    "det_H": det_H(H),
                    "allowed_x": [
                        normalize_fraction(value) for value in sorted(x_values)
                    ],
                    "tm_site_count": tm_count,
                    "na_site_count": sodium,
                    "oxygen_count": oxygen,
                    "atom_count_full": len(structure),
                }
            )
    return frameworks


def _check_tm_ratio(site_count: int, ratio: dict[str, int | float]) -> None:
    total = sum(ratio.values())
    counts = [site_count * value / total for value in ratio.values()]
    if any(abs(value - round(value)) > 1e-8 for value in counts):
        raise ValueError(f"{site_count} 个 TM 位点无法满足比例 {ratio}")

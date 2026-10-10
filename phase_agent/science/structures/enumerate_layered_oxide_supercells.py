"""Enumerate candidate in-plane supercells for layered-oxide structures."""

from __future__ import annotations

from fractions import Fraction
from typing import Any, Iterable

import numpy as np

from phase_agent.science.structures.boundary_utils import (
    det_H,
    load_structure,
    normalize_H,
    normalize_fraction,
)


# Layered-oxide recommendations only. Pass the desired subset (or custom
# matrices) through ``p_small_list``; selected matrices are conjunctive.
LAYERED_OXIDE_P_SMALL_RECOMMENDATIONS = (
    ((1, 0, 0), (0, 2, 0), (0, 0, 1)),
    ((1, 2, 0), (0, 3, 0), (0, 0, 1)),
)


def enumerate_layered_oxide_supercells(
    phase: str,
    mother_structure: Any,
    sizes: Iterable[int],
    p_small_list: Iterable[Any],
    min_distance: float = 2.0,
) -> list[dict[str, Any]]:
    """Return layered-oxide supercell candidates without changing the boundary.

    ``sizes`` are the supercell multiplicities used by icet. The returned
    ``x_list`` follows the supplied enumeration convention ``1/S ... 1`` and
    deliberately excludes zero. Every selected ``P_small`` must be contained
    in a candidate. ``H`` is returned in the project's canonical 3x3 format.
    """
    phase = str(phase).strip().upper()
    if not phase:
        raise ValueError("phase 不能为空")
    if not np.isfinite(min_distance) or min_distance < 0:
        raise ValueError("min_distance 必须是非负有限数值，单位为 Å")

    size_values = _normalize_sizes(sizes)
    p_small_matrices = _normalize_p_small_list(p_small_list)

    try:
        from pymatgen.io.ase import AseAtomsAdaptor
    except ImportError as error:  # pragma: no cover - depends on py1 setup
        raise RuntimeError("该功能需要 py1 环境中的 pymatgen 与 ASE") from error

    structure = load_structure(mother_structure)
    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.pbc = [True, True, False]
    primitive_cell = np.asarray(atoms.cell, dtype=float)

    candidates: list[dict[str, Any]] = []
    seen: set[tuple[int, tuple[int, ...]]] = set()
    for size in size_values:
        for supercell in _enumerate_icet_supercells(atoms, size):
            transformation = np.linalg.solve(
                primitive_cell.T,
                np.asarray(supercell.cell, dtype=float).T,
            ).T
            rounded = np.rint(transformation).astype(int)
            if not np.allclose(transformation, rounded, atol=1e-8, rtol=0):
                raise ValueError(f"icet 返回的 {phase}、S={size} 超胞不能映射为整数矩阵")
            if not _is_in_plane_transform(rounded):
                raise ValueError(f"icet 返回了非层内扩胞矩阵：{rounded.tolist()}")

            H = normalize_H(rounded)
            if det_H(H) != size:
                raise ValueError(f"icet 的 size 与超胞行列式不一致：S={size}, det(H)={det_H(H)}")
            if not can_contain(H, p_small_matrices):
                continue

            min_periodic_distance = _minimum_periodic_distance_2d(
                np.asarray(supercell.cell, dtype=float)
            )
            if min_periodic_distance / 2 < min_distance:
                continue

            key = (size, tuple(value for row in H for value in row))
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "phase": phase,
                    "S": size,
                    "H": H,
                    "det_H": det_H(H),
                    "R_cut": round(min_periodic_distance / 2, 1),
                    "x_list": [normalize_fraction(Fraction(i, size)) for i in range(1, size + 1)],
                }
            )

    return candidates


def can_contain(
    H_large: Any,
    p_small_list: Iterable[Any],
    tol: float = 1e-8,
) -> bool:
    """Whether ``H_large`` contains every selected smaller in-plane cell."""
    large = np.asarray(normalize_H(H_large), dtype=float)
    for raw_small in p_small_list:
        small = np.asarray(normalize_H(raw_small), dtype=float)
        multiplier = np.linalg.solve(small.T, large.T).T
        if not np.allclose(multiplier, np.rint(multiplier), atol=tol, rtol=0):
            return False
    return True


def _normalize_sizes(sizes: Iterable[int]) -> list[int]:
    if isinstance(sizes, (str, bytes)):
        raise TypeError("sizes 必须是正整数序列")
    result: list[int] = []
    seen: set[int] = set()
    try:
        values = list(sizes)
    except TypeError as error:
        raise TypeError("sizes 必须是正整数序列") from error
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"size 必须是正整数，当前为 {value!r}")
        size = int(value)
        if size <= 0:
            raise ValueError(f"size 必须大于零，当前为 {size}")
        if size not in seen:
            seen.add(size)
            result.append(size)
    if not result:
        raise ValueError("sizes 不能为空")
    return result


def _normalize_p_small_list(p_small_list: Iterable[Any]) -> list[list[list[int]]]:
    if _is_matrix(p_small_list):
        values = [p_small_list]
    else:
        try:
            values = list(p_small_list)
        except TypeError as error:
            raise TypeError("p_small_list 必须是 2×2/3×3 矩阵组成的序列") from error
        if _is_matrix(values):
            values = [values]

    result = []
    for value in values:
        matrix = normalize_H(value)
        if not _is_in_plane_transform(np.asarray(matrix, dtype=int)):
            raise ValueError(f"P_small 必须是层内 2D 矩阵：{matrix}")
        det_H(matrix)
        result.append(matrix)
    return result


def _is_matrix(value: Any) -> bool:
    try:
        return np.asarray(value, dtype=object).shape in ((2, 2), (3, 3))
    except (TypeError, ValueError):
        return False


def _is_in_plane_transform(matrix: np.ndarray) -> bool:
    return bool(
        matrix.shape == (3, 3)
        and np.array_equal(matrix[2], np.array([0, 0, 1]))
        and np.array_equal(matrix[:2, 2], np.zeros(2, dtype=int))
    )


def _enumerate_icet_supercells(atoms: Any, size: int) -> Iterable[Any]:
    try:
        from icet.tools import enumerate_supercells
    except ImportError as error:  # pragma: no cover - depends on py1 setup
        raise RuntimeError("该功能需要 py1 环境中的 icet") from error
    return enumerate_supercells(atoms, sizes=[size], niggli_reduce=False)


def _minimum_periodic_distance_2d(cell: np.ndarray) -> float:
    try:
        from ase.geometry import minkowski_reduce
    except ImportError as error:  # pragma: no cover - depends on py1 setup
        raise RuntimeError("该功能需要 py1 环境中的 ASE") from error
    temporary_cell = np.vstack([cell[:2], [0.0, 0.0, 1000.0]])
    reduced_cell, _ = minkowski_reduce(
        temporary_cell,
        pbc=[True, True, False],
    )
    return float(min(np.linalg.norm(reduced_cell[0]), np.linalg.norm(reduced_cell[1])))

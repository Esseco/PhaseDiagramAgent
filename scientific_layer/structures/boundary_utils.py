"""搜索边界、超胞矩阵和母结构的通用工具。"""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor


def validate_boundary(boundary: dict[str, Any]) -> dict[str, Any]:
    missing = {"P", "H", "TM_ratio"} - boundary.keys()
    if missing:
        raise ValueError(f"boundary 缺少字段：{sorted(missing)}")
    return json_safe(boundary)


def normalize_H(H: Any) -> list[list[int]]:
    if not isinstance(H, (list, tuple, np.ndarray)):
        raise ValueError("H 必须是 2×2 或 3×3 整数矩阵")
    shape = np.shape(H)
    if shape == (2, 2):
        expanded = np.eye(3, dtype=object)
        expanded[:2, :2] = np.asarray(H, dtype=object)
        H = expanded
    elif shape != (3, 3):
        raise ValueError(f"H 必须是 2×2 或 3×3 整数矩阵，当前 shape={shape}")
    matrix = []
    for row in H:
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in row
        ):
            raise TypeError("H 必须只包含整数")
        matrix.append([int(value) for value in row])
    return matrix


def det_H(H: Any) -> int:
    determinant = float(np.linalg.det(np.asarray(normalize_H(H), dtype=float)))
    rounded = round(determinant)
    if abs(determinant - rounded) > 1e-8 or rounded == 0:
        raise ValueError(f"H 的行列式必须是非零整数，当前为 {determinant}")
    return abs(rounded)


def allowed_phases(config: Any) -> set[str]:
    if isinstance(config, str):
        return {config.upper()}
    if isinstance(config, (list, tuple)):
        return {str(item).upper() for item in config}
    if isinstance(config, dict):
        result: set[str] = set()
        for value in config.values():
            result.update(allowed_phases(value))
        return result
    raise TypeError("boundary['P'] 必须是相名、列表或分组 dict")


def allowed_phases_at_x(config: Any, x: int | float | str | Fraction) -> set[str]:
    """Return the phases allowed at Na content x; legacy phase lists allow all x."""
    if not isinstance(config, dict) or not ({"at_x", "intermediate"} & set(config)):
        return allowed_phases(config)
    exact = config.get("at_x")
    intermediate = config.get("intermediate")
    if not isinstance(exact, dict) or not isinstance(intermediate, list):
        raise ValueError("boundary.P 需要 at_x 映射和 intermediate 相列表")
    if set(config) != {"at_x", "intermediate"}:
        raise ValueError("boundary.P 仅允许 at_x 与 intermediate 两个分组")
    if not exact or not intermediate:
        raise ValueError("boundary.P 的 at_x 和 intermediate 均须非空")
    for key, phases in exact.items():
        at_x = Fraction(normalize_fraction(key))
        if not 0 <= at_x <= 1 or not isinstance(phases, list) or not phases:
            raise ValueError("boundary.P.at_x 必须映射 [0,1] 内组分到非空相列表")
    if any(not isinstance(phase, str) or not phase.strip() for phase in intermediate):
        raise ValueError("boundary.P.intermediate 必须是非空相名列表")
    fraction = Fraction(normalize_fraction(x))
    for key, phases in exact.items():
        if Fraction(normalize_fraction(key)) == fraction:
            return allowed_phases(phases)
    return allowed_phases(intermediate) if 0 < fraction < 1 else set()


def allowed_supercells(config: Any, phase: str) -> list[list[list[int]]]:
    values = (
        config.get(phase, config.get(phase.upper(), []))
        if isinstance(config, dict)
        else config
    )
    matrices = _collect_matrices(values)
    if not matrices:
        raise ValueError(f"相 {phase!r} 没有允许的超胞矩阵")
    unique = {compact_json(normalize_H(item)): normalize_H(item) for item in matrices}
    return list(unique.values())


def load_structure(value: Structure | str | Path | Any) -> Structure:
    if isinstance(value, Structure):
        return value.copy()
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"结构文件不存在：{path}")
        return Structure.from_file(path)
    if hasattr(value, "get_chemical_symbols") and hasattr(value, "get_positions"):
        return AseAtomsAdaptor.get_structure(value)
    raise TypeError("结构必须是 pymatgen Structure、ASE Atoms 或文件路径")


def normalize_fraction(value: int | float | str | Fraction) -> str:
    if isinstance(value, bool):
        raise TypeError("分数不能是布尔值")
    try:
        fraction = value if isinstance(value, Fraction) else Fraction(str(value))
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(f"无法识别分数 {value!r}") from error
    return str(fraction.numerator) if fraction.denominator == 1 else str(fraction)


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON 数据不能包含 NaN 或无穷大")
        return value
    if isinstance(value, Fraction):
        return normalize_fraction(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in sorted(value.items(), key=lambda x: str(x[0]))
        }
    raise TypeError(f"不支持保存的数据类型：{type(value).__name__}")


def compact_json(value: Any) -> str:
    return json.dumps(
        json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _collect_matrices(value: Any) -> list[Any]:
    if isinstance(value, dict):
        if "H_matrix" in value:
            return _collect_matrices(value["H_matrix"])
        if "H" in value:
            return _collect_matrices(value["H"])
        return [matrix for item in value.values() for matrix in _collect_matrices(item)]
    if isinstance(value, np.ndarray):
        if value.shape in ((2, 2), (3, 3)):
            return [value]
        if value.ndim >= 3 and value.shape[-2:] in ((2, 2), (3, 3)):
            return list(value.reshape((-1, *value.shape[-2:])))
        raise ValueError(f"H 集合末两维必须是 2×2 或 3×3，当前 shape={value.shape}")
    if isinstance(value, (list, tuple)):
        if not value:
            return []
        if np.shape(value) in ((2, 2), (3, 3)):
            return [value]
        return [matrix for item in value for matrix in _collect_matrices(item)]
    raise TypeError(f"无法从 {type(value).__name__} 读取 H 矩阵")

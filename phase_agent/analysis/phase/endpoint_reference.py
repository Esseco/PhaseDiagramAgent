"""Shared endpoint chord subtraction; energies keep the caller's unit and source."""

import numpy as np

PLOT_RULE_VERSION = "retained-endpoint-relative-visible-v2"


def endpoint_relative_energies(x, energy, *, reference=None):
    """Use one endpoint pair for all phases; an explicit pair also transforms crosses."""
    x, energy = np.asarray(x), np.asarray(energy)
    if np.iscomplexobj(x) or np.iscomplexobj(energy):
        raise ValueError("端点归一化要求实数")
    x, energy = np.asarray(x, dtype=float), np.asarray(energy, dtype=float)
    if x.ndim != 1 or energy.shape != x.shape or not x.size:
        raise ValueError("端点归一化要求等长非空一维数据")
    if not np.isfinite(x).all() or not np.isfinite(energy).all():
        raise ValueError("端点归一化要求有限组成和能量")
    if reference is None:
        left, right = float(x.min()), float(x.max())
        reference = {
            "x": [left, right],
            "energy": [float(energy[x == left].min()), float(energy[x == right].min())],
        }
    bounds = np.asarray(reference["x"], dtype=float)
    ends = np.asarray(reference["energy"], dtype=float)
    if (
        bounds.shape != (2,)
        or ends.shape != (2,)
        or not np.isfinite(bounds).all()
        or not np.isfinite(ends).all()
        or bounds[0] >= bounds[1]
        or x.min() < bounds[0] - 1e-9
        or x.max() > bounds[1] + 1e-9
    ):
        raise ValueError("缺少两个有效端点，或数据超出端点覆盖范围")
    # NumPy supplies the same linear interpolation used by the local phase-diagram library.
    relative = energy - np.interp(x, bounds, ends)
    return relative.tolist(), {"x": bounds.tolist(), "energy": ends.tolist()}

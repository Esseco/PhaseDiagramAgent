"""Use the project's explicit phase-at-composition rule, without chemistry defaults."""

from copy import deepcopy
from fractions import Fraction

from phase_agent.science.structures.boundary_utils import allowed_phases_at_x


def phase_boundary_rule(state):
    rule = (
        (((state or {}).get("confirmed_config") or {}).get("system") or {}).get("boundary") or {}
    ).get("P")
    return deepcopy(rule) if isinstance(rule, dict) and "at_x" in rule else None


def outside_phase_boundary(row, rule):
    if rule is None:
        return False
    composition = row.get("composition") or {}
    oxygen = Fraction(str(composition.get("O", 0)))
    if oxygen <= 0:
        raise ValueError("相图记录缺少有效氧组成")
    x = 2 * Fraction(str(composition.get("Na", 0))) / oxygen
    return str(row.get("phase") or "").upper() not in allowed_phases_at_x(rule, x)


def assert_snapshot_boundary(state, snapshot):
    rule = phase_boundary_rule(state)
    count = sum(outside_phase_boundary(row, rule) for row in snapshot.get("entries") or [])
    if count:
        raise ValueError(
            f"已保存相图有{count}条记录超出当前相/Na含量边界，须先重建相图；不静默沿用旧凸包"
        )

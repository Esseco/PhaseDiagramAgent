"""Expose legal initial phase/composition cells as metadata, not a sampling target."""

from fractions import Fraction
import json

from pymatgen.core import Element

from phase_agent.science.structures import boundary_utils as utils

SAMPLING_SCALE_INSTRUCTION = (
    "采样规模须充分：空池首轮和较广补充需求应优先评估数百候选的方案，"
    "按相、Na区间、扩胞多样性及每branch初态成本说明数量依据。"
    "不把每区域十几个、全轮几十个作为保守默认；仅请求几十个时，须说明有限合法空间、"
    "已有覆盖、明确局部需求或资源约束等具体理由。"
    "后续补充同样按缺口广度与收益决定规模，不机械沿用小批数量。"
    "数百是规模评估方向，不是固定配额或最低数量；每轮上限1000，不凑数、不穷举。"
)

INSTRUCTION = (
    "首轮优先覆盖用户选定的相、扩胞和Na范围；按需求采样，每轮最多生成1000个branch。"
    "合法组合仅用于范围与覆盖统计，不要求穷举或把数量作为目标。"
    "Na=0豁免层占位，其他含量须满足各Na层占位约束；未采到的区域如实报告。"
    + SAMPLING_SCALE_INSTRUCTION
)


def cell_key(phase, matrix, x):
    return phase, json.dumps(matrix, separators=(",", ":")), str(Fraction(str(x)))


def contract(manager, references, config):
    if manager is None or manager.data.get("branches") or manager.data.get("structures"):
        return {"required": False}
    boundary = manager.boundary
    system = (
        config.get("system_config")
        or config.get("system")
        or manager.data.get("system_config")
        or {}
    )
    space = system.get("configuration_space") or {}
    roles, fixed = space.get("roles") or {}, space.get("fixed_values") or {}
    limits = (config.get("budgets") or {}).get("structure_limits") or {}
    frameworks = []
    for phase in sorted(utils.allowed_phases(boundary["P"])):
        if roles.get("P") == "fixed" and phase != str(fixed["P"]).upper():
            continue
        reference = utils.load_structure(references[phase])
        oxygen = sum(s.is_ordered and s.specie.symbol == "O" for s in reference)
        sodium = sum(s.is_ordered and s.specie.symbol == "Na" for s in reference)
        tm_sites = sum(
            site.is_ordered and Element(site.specie.symbol).is_transition_metal
            for site in reference
        )
        ratio = boundary.get("TM_ratio") or {}
        if not oxygen:
            raise ValueError("母结构缺少氧")
        for matrix in utils.allowed_supercells(boundary["H"], phase):
            size = utils.det_H(matrix)
            if ratio and any(
                abs(
                    tm_sites * size * v / sum(ratio.values())
                    - round(tm_sites * size * v / sum(ratio.values()))
                )
                > 1e-8
                for v in ratio.values()
            ):
                continue
            if roles.get("H") == "fixed" and matrix != fixed["H"]:
                continue
            if limits.get("max_det_H") is not None and size > limits["max_det_H"]:
                continue
            if limits.get("max_atoms") is not None and len(reference) * size > limits["max_atoms"]:
                continue
            from phase_agent.science.structures.rank_na_orderings_by_electrostatics import (
                count_supercell_na_layers,
            )

            layer_count = count_supercell_na_layers(reference, matrix)
            xs = [Fraction(2 * n, oxygen * size) for n in range(sodium * size + 1)]
            xs = [
                x
                for x in xs
                if 0 <= x <= 1
                and (x == 0 or x * oxygen * size / 2 >= layer_count)
                and phase in utils.allowed_phases_at_x(boundary["P"], x)
                and (roles.get("x") != "fixed" or x == Fraction(str(fixed["x"])))
            ]
            if xs:
                frameworks.append(
                    {"P": phase, "H": matrix, "det_H": size, "allowed_x": [str(x) for x in xs]}
                )
    return {
        "required": True,
        "allowed_strategies": ["coverage"],
        "instruction": INSTRUCTION,
        "frameworks": frameworks,
    }


def review(params, facts):
    if not facts.get("required"):
        return [], {"required": False}
    regions = params.get("regions") or []
    if not isinstance(regions, list) or any(not isinstance(row, dict) for row in regions):
        return ["initial_coverage_requires_regions"], {"required": True}
    if any(row.get("strategy") != "coverage" for row in regions):
        return ["initial_coverage_requires_coverage"], {"required": True}
    frameworks = facts.get("frameworks")
    if frameworks is None:
        return [], {"required": True}
    selected, actual = set(), set()
    for row in regions:
        for f in frameworks:
            if f["P"] != row.get("phase") or not row.get("min_det_H", 1) <= f["det_H"] <= row.get(
                "max_det_H", 0
            ):
                continue
            if params.get("max_det_H") is not None and f["det_H"] > params["max_det_H"]:
                continue
            if row.get("supercells") is not None and f["H"] not in row["supercells"]:
                continue
            selected.add(json.dumps(f["H"], separators=(",", ":")))
            for x in f["allowed_x"]:
                if row.get("na_min", 0) <= float(Fraction(x)) <= row.get("na_max", 1):
                    actual.add(cell_key(f["P"], f["H"], x))
    problems = []
    if not selected:
        problems.append("initial_coverage_requires_selected_supercells")
    expected = actual
    return problems, {
        "required": True,
        "expected_cells": len(expected),
        "selected_supercells": len(selected),
    }

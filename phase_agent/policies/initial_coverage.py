"""Initial coverage is a complete phase/composition grid on chosen supercells."""

from fractions import Fraction
import json

from pymatgen.core import Element

from phase_agent.science.structures import boundary_utils as utils

INSTRUCTION = (
    "首轮没有父branch，只允许coverage。先从确认边界选择一组扩胞，再覆盖这些扩胞下全部合法相和整数Na组分；"
    "中间组分的竞争相必须同时覆盖，端点按boundary.P过滤。不能用不同相的局部Na区间代替首轮覆盖。"
    "Na数量=x*氧数量/2，P3候选位点数不改变组分定义。数量由所选扩胞和整数占据决定；"
    "资源不足应在审批前缩小扩胞选择，执行被截断不能宣称覆盖完成或准备下游输入。"
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
            xs = [Fraction(2 * n, oxygen * size) for n in range(sodium * size + 1)]
            xs = [
                x
                for x in xs
                if 0 <= x <= 1
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
    expected = {
        cell_key(f["P"], f["H"], x)
        for f in frameworks
        if json.dumps(f["H"], separators=(",", ":")) in selected
        for x in f["allowed_x"]
    }
    missing = sorted(expected - actual)
    problems = []
    if not selected:
        problems.append("initial_coverage_requires_selected_supercells")
    if missing:
        problems.append(
            f"initial_coverage_missing_cells:{len(missing)}; examples={missing[:4]}; cover all legal phases/x on chosen supercells"
        )
    from phase_agent.policies.region_sampling import CONTROLS

    limit = min(
        params.get("max_new_branches", CONTROLS["review_after_new_branches"]),
        CONTROLS["review_after_new_branches"],
    )
    if len(expected) > limit:
        problems.append(
            f"initial_coverage_exceeds_branch_guard:{len(expected)}>{limit}; choose fewer supercells"
        )
    return problems, {
        "required": True,
        "expected_cells": len(expected),
        "selected_supercells": len(selected),
    }

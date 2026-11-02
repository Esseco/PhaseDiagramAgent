"""Build an MC Na-site template from a reference cell, never from Relax output."""

from fractions import Fraction

import numpy as np

from phase_agent.science.structures.boundary_utils import (
    allowed_supercells,
    load_structure,
    normalize_H,
    normalize_fraction,
)
from phase_agent.science.structures.generate_branch_structure import _assign_tm
from phase_agent.science.structures.identify_branch import extract_T


def build_mc_full_na_template(branch, boundary, phase_references, *, config=None):
    phase = str(branch["P"]).upper()
    references = {str(key).upper(): value for key, value in phase_references.items()}
    system = (config or {}).get("system") or (config or {}).get("system_config") or {}
    templates = (
        system.get("mc_full_na_templates") or (config or {}).get("mc_full_na_templates") or {}
    )
    template = templates.get(phase) or templates.get(branch["P"])
    source = template or references.get(phase)
    if source is None:
        raise ValueError(f"branch {branch.get('branch_id')}: 相 {phase} 缺少母结构/满 Na 位点模板")
    parent = load_structure(source)
    if template and phase in references:
        reference = load_structure(references[phase])
        if not np.allclose(parent.lattice.matrix, reference.lattice.matrix, atol=1e-5, rtol=0):
            raise ValueError(
                f"branch {branch.get('branch_id')}: 满 Na 模板与 {phase} 母结构晶格不一致，缺少明确位点映射"
            )

        def framework_sites(structure):
            return sorted(
                (
                    site.specie.symbol,
                    tuple(round(float(value % 1), 6) for value in site.frac_coords),
                )
                for site in structure
                if site.is_ordered and site.specie.symbol != "Na"
            )

        if framework_sites(parent) != framework_sites(reference):
            raise ValueError(
                f"branch {branch.get('branch_id')}: 满 Na 模板与 {phase} 母结构框架位点不一致"
            )
    matrix = normalize_H(branch["H"])
    if matrix not in allowed_supercells(boundary["H"], phase):
        raise ValueError(
            f"branch {branch.get('branch_id')}: H={matrix!r} 不在相 {phase!r} 的允许范围"
        )
    parent.make_supercell(matrix)
    na_count = sum(site.is_ordered and site.specie.symbol == "Na" for site in parent)
    oxygen_count = sum(site.is_ordered and site.specie.symbol == "O" for site in parent)
    if na_count == 0:
        raise ValueError(
            f"branch {branch.get('branch_id')}: 相 {phase} 母结构没有 Na 位点（候选位点池为空）；"
            "请显式配置 system.mc_full_na_templates 中该相的位点模板"
        )
    if oxygen_count == 0:
        raise ValueError(f"branch {branch.get('branch_id')}: 相 {phase} 母结构没有 O 位点")
    target = Fraction(normalize_fraction(branch["x"])) * oxygen_count / 2
    if target.denominator != 1 or not 0 <= target <= na_count:
        raise ValueError(
            f"branch {branch.get('branch_id')}: x={branch.get('x')} 需要 {target} 个 Na，"
            f"但模板提供 {na_count} 个候选 Na 位点"
        )
    space = system.get("configuration_space") or (config or {}).get("configuration_space") or {}
    fixed_t = (space.get("roles") or {}).get("T") == "fixed"
    if fixed_t:
        extract_T(parent, boundary["TM_ratio"])
    else:
        if branch.get("T") is None:
            raise ValueError(
                f"branch {branch.get('branch_id')}: 可变 T 的 MC 满 Na 模板缺少 branch.T"
            )
        _assign_tm(parent, boundary["TM_ratio"], seed=None, requested_T=list(branch["T"]))
        actual_t, _ = extract_T(parent, boundary["TM_ratio"])
        if actual_t != list(branch["T"]):
            raise ValueError(f"branch {branch.get('branch_id')}: 模板 TM 占位与 branch.T 不一致")
    parent.sort()
    return parent

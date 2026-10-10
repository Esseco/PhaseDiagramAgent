"""Build an MC Na-site template from a reference cell, never from Relax output."""

import numpy as np

from phase_agent.science.structures.boundary_utils import load_structure
from phase_agent.science.structures.generate_branch_structure import generate_branch_structure
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
    na_count = sum(site.is_ordered and site.specie.symbol == "Na" for site in parent)
    oxygen_count = sum(site.is_ordered and site.specie.symbol == "O" for site in parent)
    if na_count == 0:
        raise ValueError(
            f"branch {branch.get('branch_id')}: 相 {phase} 母结构没有 Na 位点；"
            "请显式配置 system.mc_full_na_templates 中该相的位点模板"
        )
    if oxygen_count != 2 * na_count:
        raise ValueError(
            f"branch {branch.get('branch_id')}: 满 Na 模板须满足 Na:TMO2 位点数；"
            f"母结构有 {na_count} 个 Na、{oxygen_count} 个 O"
        )
    space = system.get("configuration_space") or (config or {}).get("configuration_space") or {}
    fixed_t = (space.get("roles") or {}).get("T") == "fixed"
    if not fixed_t and branch.get("T") is None:
        raise ValueError(f"branch {branch.get('branch_id')}: 可变 T 的 MC 满 Na模板缺少 branch.T")
    structure = generate_branch_structure(
        boundary,
        phase=phase,
        H=branch["H"],
        x=1,
        T=None if fixed_t else list(branch["T"]),
        preserve_reference_tm=fixed_t,
        phase_references={phase: source},
        enforce_phase_composition=False,
    )
    actual_t, _ = extract_T(structure, boundary["TM_ratio"])
    if branch.get("T") is not None and actual_t != list(branch["T"]):
        raise ValueError(f"branch {branch.get('branch_id')}: 模板 TM 占位与 branch.T 不一致")
    return structure

"""Expand the user's layered-oxide H bounds into the existing explicit H boundary."""

from copy import deepcopy

from phase_agent.science.structures.boundary_utils import allowed_phases
from phase_agent.science.structures.enumerate_layered_oxide_supercells import (
    LAYERED_OXIDE_P_SMALL_RECOMMENDATIONS,
    enumerate_layered_oxide_supercells,
)


def materialize_layered_h(config: dict) -> dict:
    result = deepcopy(config)
    system = result.get("system") or {}
    settings = system.get("H_generation")
    if not settings or not settings.get("enabled"):
        return result
    boundary = system.get("boundary") or {}
    if not boundary.get("P") or not boundary.get("TM_ratio"):
        raise ValueError(
            "请先设置科学边界 system.boundary.P 和 TM_ratio；模板不代表用户选择，未生成 H"
        )
    from phase_agent.configuration.schema.validate_configuration_space import (
        validate_configuration_space,
    )

    role_check = validate_configuration_space(system)
    if not role_check["valid"]:
        raise ValueError("; ".join(role_check["errors"]))
    if (system.get("boundary") or {}).get("H"):
        raise ValueError("H_generation 已启用时 boundary.H 应留空；显式 H 请先关闭 H_generation")
    if system.get("system_id") != "layered_na_tm_oxide":
        raise ValueError("H_generation 当前仅支持层氧体系")
    lower, upper, step = (settings.get(key) for key in ("size_min", "size_max", "size_step"))
    if any(type(value) is not int or value <= 0 for value in (lower, upper, step)) or lower > upper:
        raise ValueError("H_generation 的 size_min/size_max/size_step 必须是有效正整数范围")
    indices = settings.get("selected_recommendation_indices") or []
    if not isinstance(indices, list) or any(
        type(i) is not int or i < 0 or i >= len(LAYERED_OXIDE_P_SMALL_RECOMMENDATIONS)
        for i in indices
    ):
        raise ValueError("selected_recommendation_indices 索引无效")
    selected = [LAYERED_OXIDE_P_SMALL_RECOMMENDATIONS[i] for i in indices]
    selected.extend(settings.get("additional_containment_matrices") or [])
    if not selected:
        raise ValueError("请在 H_generation 中选择至少一个推荐包含矩阵，或填写自定义矩阵")
    boundary = system.get("boundary") or {}
    phases = sorted(allowed_phases(boundary.get("P")))
    references = system.get("phase_references") or {}
    generated = {}
    for phase in phases:
        if phase not in references:
            raise ValueError(f"缺少相 {phase} 的母结构，无法生成 H")
        try:
            candidates = enumerate_layered_oxide_supercells(
                phase,
                references[phase],
                range(lower, upper + 1, step),
                selected,
                min_distance=settings.get("min_distance_angstrom", 2.0),
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise ValueError(
                f"相 {phase} 的 H 生成失败；母结构={references[phase]}，"
                f"size={lower}..{upper} 步长 {step}，原因：{error}"
            ) from error
        matrices = [item["H"] for item in candidates]
        if not matrices:
            raise ValueError(
                f"相 {phase} 没有合法 H；母结构={references[phase]}，"
                f"size={lower}..{upper} 步长 {step}，"
                f"最小周期半径={settings.get('min_distance_angstrom', 2.0)} Å"
            )
        generated[phase] = matrices
    boundary["H"] = generated
    system["boundary"] = boundary
    constraints = system.setdefault("constraints", {})
    constraints["phases"] = phases
    constraints["TM_ratio"] = deepcopy(boundary.get("TM_ratio"))
    result["system"] = system
    from phase_agent.configuration.session.synchronize_system_scope import synchronize_system_scope

    return synchronize_system_scope(result)

"""Give the configuration agent compact facts verified by local code."""

from __future__ import annotations

from pathlib import Path
from phase_agent.decisions.agent.generation_plan import configured_generation_strategies

from phase_agent.science.structures.boundary_utils import allowed_phases, det_H


def summarize_config_for_agent(
    config: dict, *, source_path, source_hash, generated_h=False
) -> dict:
    system = config.get("system") or {}
    boundary = system.get("boundary") or {}
    phases = sorted(allowed_phases(boundary["P"])) if boundary.get("P") else []
    h = boundary.get("H") or {}
    references = system.get("phase_references") or {}

    def reference_path(phase):
        value = references.get(phase)
        return value.get("path") if isinstance(value, dict) else value

    h_counts = {}
    for phase in phases:
        matrices = h.get(phase)
        if isinstance(matrices, list):
            determinants = [det_H(matrix) for matrix in matrices]
            h_counts[phase] = {
                "count": len(matrices),
                "min_det": min(determinants) if determinants else None,
                "max_det": max(determinants) if determinants else None,
            }
    from phase_agent.configuration.session.split_project_config import (
        read_document,
        run_config_path,
    )

    try:
        document = read_document(source_path)
        run_source = (
            str(run_config_path(source_path, document)) if "run_config_file" in document else None
        )
    except (OSError, ValueError):
        run_source = None
    return {
        "initial_config_path": str(source_path),
        "run_config_path": run_source,
        "source_path": str(source_path),
        "source_sha256": source_hash,
        "boundary_P": boundary.get("P"),
        "TM_ratio": boundary.get("TM_ratio"),
        "configuration_space": system.get("configuration_space"),
        "enabled_generation_strategies": configured_generation_strategies(config),
        "allowed_phases_union": phases,
        "boundary_H_keys": sorted(h),
        "generated_H": bool(generated_h),
        "H_by_phase": h_counts,
        "H_generation": system.get("H_generation"),
        "phase_references": {
            phase: {
                "path": str(reference_path(phase) or ""),
                "exists": Path(str(reference_path(phase) or "")).is_file(),
            }
            for phase in phases
        },
        "constraints_phases": (system.get("constraints") or {}).get("phases"),
        "mlip_version": (config.get("calculation") or {}).get("mlip_version"),
        "model_path": (config.get("mlip") or {}).get("model_path"),
        "budget_max_det_H": ((config.get("budgets") or {}).get("structure_limits") or {}).get(
            "max_det_H"
        ),
        "qbc_max_det_H": (
            (
                ((config.get("qbc") or {}).get("budget_limits") or {}).get("structure_limits") or {}
            ).get("max_det_H")
        ),
        "scheduler_configured": bool(
            ((config.get("supercomputer") or {}).get("scheduler") or {}).get("submit_command")
        ),
        "scheduler_required_for_advice": False,
    }


def format_scientific_scope(config):
    """Show the exact problem definition before asking for configuration approval."""
    import json

    system = config.get("system") or {}
    boundary = system.get("boundary") or {}
    space = system.get("configuration_space") or {}
    roles = space.get("roles") or {}
    ratio = boundary.get("TM_ratio") or {}
    role_labels = {"fixed": "固定", "branch": "可变", "internal": "内部搜索"}
    variables = ", ".join(
        f"{name}={role_labels.get(roles.get(name), '未设置')}" for name in ("H", "P", "x", "T", "N")
    )
    lines = [
        "科学边界（确认前请核对）：",
        "P="
        + json.dumps(boundary.get("P"), ensure_ascii=False)
        + "; TM_ratio="
        + json.dumps(ratio, ensure_ascii=False),
        variables
        + "; fixed_values="
        + json.dumps(space.get("fixed_values") or {}, ensure_ascii=False),
        "合法生成策略=" + ", ".join(configured_generation_strategies(config)),
    ]
    if len(ratio) == 1:
        lines.append("单 TM 元素：无 TM 交换/排列自由度；保留正常原子相互作用与弛豫。")
    if len(allowed_phases(boundary.get("P") or [])) == 1 or roles.get("P") == "fixed":
        lines.append("不搜索其他相，不分配竞争相策略。")
    if roles.get("T") == "fixed":
        lines.append(
            "TM 排布来源=" + str(space.get("fixed_T_source")) + "；固定占位不等于冻结原子坐标。"
        )
    return "\n".join(lines)

"""Give the configuration agent compact facts verified by local code."""

from __future__ import annotations

from pathlib import Path

from scientific_layer.structures.boundary_utils import allowed_phases, det_H


def summarize_config_for_agent(config: dict, *, source_path, source_hash,
                               generated_h=False) -> dict:
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
    return {
        "source_path": str(source_path),
        "source_sha256": source_hash,
        "boundary_P": boundary.get("P"),
        "allowed_phases_union": phases,
        "boundary_H_keys": sorted(h),
        "generated_H": bool(generated_h),
        "H_by_phase": h_counts,
        "H_generation": system.get("H_generation"),
        "phase_references": {
            phase: {"path": str(reference_path(phase) or ""),
                    "exists": Path(str(reference_path(phase) or "")).is_file()}
            for phase in phases
        },
        "constraints_phases": (system.get("constraints") or {}).get("phases"),
        "mlip_version": (config.get("calculation") or {}).get("mlip_version"),
        "model_path": (config.get("mlip") or {}).get("model_path"),
        "budget_max_det_H": ((config.get("budgets") or {}).get("structure_limits") or {}).get("max_det_H"),
        "qbc_max_det_H": ((((config.get("qbc") or {}).get("budget_limits") or {})
                            .get("structure_limits") or {}).get("max_det_H")),
        "scheduler_configured": bool(((config.get("supercomputer") or {}).get("scheduler") or {})
                                     .get("submit_command")),
        "scheduler_required_for_advice": False,
    }

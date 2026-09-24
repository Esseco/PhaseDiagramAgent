"""固定 branch 参数，通过可续算后端分段搜索 V。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from scientific_layer.structures.identify_branch import identify_branch_parameters


def run_mlip_mc(
    structure: Any,
    branch: dict[str, Any],
    boundary: dict[str, Any],
    phase_references: dict[str, Any],
    *,
    segment_budget: int,
    backend: Callable[..., dict[str, Any]] | None = None,
    model_path: str | Path | None = None,
    parameters: dict[str, Any] | None = None,
    checkpoint: str | Path | None = None,
    work_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Run one independent MC segment; a structure restart is not a full-chain resume."""
    if (
        isinstance(segment_budget, bool)
        or not isinstance(segment_budget, int)
        or segment_budget <= 0
    ):
        raise ValueError("segment_budget 必须是正整数")
    supplied = dict(parameters or {})
    patience = supplied.get("patience_steps")
    min_improvement = supplied.get("min_improvement")
    contract = {"max_mc_steps": segment_budget, "requested_max_mc_steps": segment_budget,
                "patience_steps": patience, "min_improvement": min_improvement,
                "requested_budget": segment_budget,
                "restart_mode": "new_segment_from_structure", "strict_chain_resume": False}
    if backend is None or model_path is None:
        return {
            "stage": "deep_search",
            "status": "not_configured",
            "converged": None,
            "outputs": {},
            "checkpoint": str(checkpoint) if checkpoint else None,
            "actual_cost": None,
            "error": {
                "type": "NotConfigured",
                "message": "MC backend 或 MLIP model_path 未配置",
            }, **contract,
        }
    try:
        result = backend(
            structure=structure,
            branch=branch,
            model_path=Path(model_path),
            segment_budget=segment_budget,
            checkpoint=Path(checkpoint) if checkpoint else None,
            parameters=supplied,
            work_directory=Path(work_directory) if work_directory else None,
        )
        candidate = result.get("structure")
        if candidate is not None:
            found = identify_branch_parameters(
                candidate, boundary, phase_references=phase_references
            )
            changed = [key for key in ("P", "H", "x", "T") if found[key] != branch[key]]
            if changed:
                raise ValueError(f"MC 改变了固定 branch 参数：{changed}")
        actual_steps = result.get("actual_mc_steps", result.get("steps"))
        stop_reason = result.get("stop_reason") or ("patience" if result.get("early_stopped") else
                                                     "max_steps" if actual_steps == segment_budget else "backend_reported")
        return {
            "stage": "deep_search",
            "status": "completed"
            if result.get("segment_complete", True)
            else "running",
            "converged": result.get("converged"),
            "structure": candidate,
            "outputs": result,
            "checkpoint": str(result.get("checkpoint"))
            if result.get("checkpoint")
            else None,
            "actual_cost": result.get("actual_cost"),
            "actual_gpu_core_hours": result.get("actual_gpu_core_hours", result.get("gpu_core_hours")),
            "job_accounting": result.get("job_accounting"),
            "actual_mc_steps": actual_steps, "stop_reason": stop_reason,
            **contract,
            "error": None,
        }
    except Exception as error:
        return {
            "stage": "deep_search",
            "status": "failed",
            "converged": False,
            "outputs": {},
            "checkpoint": str(checkpoint) if checkpoint else None,
            "actual_cost": None,
            "actual_gpu_core_hours": None,
            "estimated_cost": None,
            "error": {"type": type(error).__name__, "message": str(error)}, **contract,
        }

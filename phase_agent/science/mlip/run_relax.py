"""通过显式配置的后端运行 MLIP 弛豫。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


def run_mlip_relax(
    structure: Any,
    *,
    backend: Callable[..., dict[str, Any]] | None = None,
    model_path: str | Path | None = None,
    parameters: dict[str, Any] | None = None,
    work_directory: str | Path | None = None,
) -> dict[str, Any]:
    """后端应返回 structure、energy、converged、cost 等真实结果。"""
    if backend is None:
        return _missing("MLIP relax backend 未配置")
    if model_path is None:
        return _missing("MLIP model_path 未配置")
    try:
        result = backend(
            structure=structure,
            model_path=Path(model_path),
            parameters=dict(parameters or {}),
            work_directory=Path(work_directory) if work_directory else None,
        )
        return {
            "stage": "relax_and_feature",
            "status": "completed",
            "converged": result.get("converged"),
            "structure": result.get("structure"),
            "outputs": result,
            "actual_cost": result.get("cost"),
            "error": None,
        }
    except Exception as error:
        return _failed(error)


def _missing(message):
    return {
        "stage": "relax_and_feature",
        "status": "not_configured",
        "converged": None,
        "outputs": {},
        "actual_cost": None,
        "error": {"type": "NotConfigured", "message": message},
    }


def _failed(error):
    return {
        "stage": "relax_and_feature",
        "status": "failed",
        "converged": False,
        "outputs": {},
        "actual_cost": None,
        "error": {"type": type(error).__name__, "message": str(error)},
    }

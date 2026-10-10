"""通过显式配置的后端运行 DFT 弛豫。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from phase_agent.science.dft.create_atomate_workflow import (
    generate_dft_workflow_with_atomate,
)


def run_dft_relax(
    structure: Any,
    *,
    backend: Callable[..., dict[str, Any]] | None = None,
    parameters: dict[str, Any] | None = None,
    work_directory: str | Path | None = None,
    restart_from: str | Path | None = None,
) -> dict[str, Any]:
    """优先使用显式 backend，否则尝试 atomate 构建 DFT 工作流。"""
    if backend is None and work_directory is not None:
        backend = _atomate_relax
    if backend is None:
        return {
            "stage": "dft_relax",
            "status": "not_configured",
            "converged": None,
            "outputs": {},
            "actual_cost": None,
            "error": {"type": "NotConfigured", "message": "DFT relax backend 未配置"},
        }
    try:
        result = backend(
            structure=structure,
            parameters=dict(parameters or {}),
            work_directory=Path(work_directory) if work_directory else None,
            restart_from=Path(restart_from) if restart_from else None,
        )
        return {
            "stage": "dft_relax",
            "status": result.get("status", "completed"),
            "converged": result.get("converged"),
            "structure": result.get("structure"),
            "outputs": result,
            "actual_cost": result.get("cost"),
            "result_path": result.get("calculation_directory"),
            "error": result.get("error"),
        }
    except Exception as error:
        return {
            "stage": "dft_relax",
            "status": "failed",
            "converged": False,
            "outputs": {},
            "actual_cost": None,
            "error": {"type": type(error).__name__, "message": str(error)},
        }


def _atomate_relax(*, structure, parameters, work_directory, restart_from=None):
    config = dict(parameters or {})
    if restart_from is not None:
        config["restart_from"] = str(restart_from)
    return generate_dft_workflow_with_atomate(
        structure,
        calculation_type="relax",
        work_directory=work_directory,
        parameters=config,
    )

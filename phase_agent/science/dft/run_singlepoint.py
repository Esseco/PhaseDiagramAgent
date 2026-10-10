"""通过显式配置的后端运行 DFT 单点。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from phase_agent.science.dft.create_atomate_workflow import (
    generate_dft_workflow_with_atomate,
)


def run_dft_singlepoint(
    structure: Any,
    *,
    backend: Callable[..., dict[str, Any]] | None = None,
    parameters: dict[str, Any] | None = None,
    work_directory: str | Path | None = None,
) -> dict[str, Any]:
    """优先使用显式 backend，否则尝试 atomate 构建 DFT 工作流。"""
    if backend is None and work_directory is not None:
        backend = _atomate_singlepoint
    return _run("dft_single_point", structure, backend, parameters, work_directory)


def _atomate_singlepoint(*, structure, parameters, work_directory):
    return generate_dft_workflow_with_atomate(
        structure,
        calculation_type="singlepoint",
        work_directory=work_directory,
        parameters=parameters,
    )


def _run(stage, structure, backend, parameters, work_directory):
    if backend is None:
        return {
            "stage": stage,
            "status": "not_configured",
            "converged": None,
            "outputs": {},
            "actual_cost": None,
            "error": {
                "type": "NotConfigured",
                "message": "DFT single-point backend 未配置",
            },
        }
    try:
        result = backend(
            structure=structure,
            parameters=dict(parameters or {}),
            work_directory=Path(work_directory) if work_directory else None,
        )
        status = result.get("status", "completed")
        return {
            "stage": stage,
            "status": status,
            "converged": result.get("converged"),
            "outputs": result,
            "actual_cost": result.get("cost"),
            "result_path": result.get("calculation_directory"),
            "error": result.get("error"),
        }
    except Exception as error:
        return {
            "stage": stage,
            "status": "failed",
            "converged": False,
            "outputs": {},
            "actual_cost": None,
            "error": {"type": type(error).__name__, "message": str(error)},
        }

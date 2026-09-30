"""使用 atomate 构建 DFT 工作流；本模块不启动正式计算。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable


def generate_dft_workflow_with_atomate(
    structure: Any,
    *,
    calculation_type: str,
    work_directory: str | Path,
    parameters: dict[str, Any] | None = None,
    workflow_factory: Callable[..., Any] | None = None,
    input_writer: Callable[[Any, Path], Any] | None = None,
) -> dict[str, Any]:
    """构建 atomate workflow，并把其中的 VASP 写入任务物化到本地目录。"""
    config = dict(parameters or {})
    atomate_kwargs = dict(config.pop("atomate_kwargs", {}))
    if workflow_factory is None:
        try:
            if calculation_type == "singlepoint":
                from atomate.vasp.workflows.base.core import get_wf_static

                workflow_factory = get_wf_static
            elif calculation_type == "relax":
                from atomate.vasp.workflows.base.core import (
                    get_wf_structure_optimization,
                )

                workflow_factory = get_wf_structure_optimization
            else:
                raise ValueError(f"未知 DFT 计算类型：{calculation_type}")
        except (ImportError, ModuleNotFoundError) as error:
            return {
                "status": "not_configured",
                "error": f"atomate DFT 工作流依赖不完整：{error}",
            }
    directory = Path(work_directory)
    directory.mkdir(parents=True, exist_ok=True)
    workflow = workflow_factory(structure, **atomate_kwargs)
    workflow_path = directory / "atomate_workflow.json"
    payload = workflow.to_dict() if hasattr(workflow, "to_dict") else workflow
    workflow_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    writer = input_writer or _materialize_vasp_inputs
    try:
        written = writer(workflow, directory)
        incar_path = directory / "INCAR"
        if incar_path.is_file():
            from pymatgen.io.vasp.inputs import Incar
            from scientific_layer.dft.incar_policy import layered_oxide_incar
            incar = Incar.from_file(incar_path)
            incar.update(layered_oxide_incar())
            incar.write_file(incar_path)
    except Exception as error:
        return {
            "status": "failed",
            "generator": "atomate",
            "calculation_type": calculation_type,
            "calculation_directory": str(directory),
            "workflow_path": str(workflow_path),
            "error": f"atomate VASP 输入物化失败：{type(error).__name__}: {error}",
        }
    return {
        "status": "pending",
        "generator": "atomate",
        "calculation_type": calculation_type,
        "calculation_directory": str(directory),
        "workflow_path": str(workflow_path),
        "workflow_name": getattr(workflow, "name", None),
        "input_files": written,
    }


def _materialize_vasp_inputs(workflow, directory: Path):
    """Run only atomate's input-writing Firetask, never its execution tasks."""
    fireworks = list(getattr(workflow, "fws", []) or [])
    for firework in fireworks:
        for task in list(getattr(firework, "tasks", []) or []):
            name = type(task).__name__
            if name not in {"WriteVaspFromIOSet", "WriteVaspInput"}:
                continue
            previous = Path.cwd()
            try:
                os.chdir(directory)
                task.run_task(dict(getattr(firework, "spec", {}) or {}))
            finally:
                os.chdir(previous)
            required = ["INCAR", "POSCAR", "KPOINTS", "POTCAR"]
            missing = [name for name in required if not (directory / name).is_file()]
            if missing:
                raise RuntimeError(f"atomate 未生成输入文件：{missing}")
            return [str(directory / name) for name in required]
    raise RuntimeError("atomate workflow 中没有找到 VASP 输入写入任务")

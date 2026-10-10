"""Build the standard dispatcher used by the search pipeline."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from phase_agent.tools.dispatch.create_default_stage_registry import create_default_stage_registry
from phase_agent.tools.dispatch.create_registry_dispatcher import create_registry_dispatcher


def create_pipeline_dispatcher(
    manager: Any,
    phase_references: dict[str, Any],
    config: dict[str, Any],
    *,
    backends: dict[str, Callable[..., dict[str, Any]]] | None = None,
    work_directory: str | Path = "outputs/tasks",
    registry: Any | None = None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create a dispatcher backed by the project stage registry.

    ``backends`` accepts either stage names or the explicit keys
    ``mlip_backend``, ``mc_backend``, ``dft_singlepoint_backend`` and
    ``dft_relax_backend``. Missing backends retain the existing stage behavior:
    MLIP/MC return ``not_configured`` while DFT may build an atomate workflow.
    """
    stage_registry = registry or create_default_stage_registry()
    root = Path(work_directory)
    configured_backends = dict(backends or {})

    def context_factory(task: dict[str, Any]) -> dict[str, Any]:
        structure_id = task.get("structure_id") or task.get("object_id")
        structures = manager.data.get("structures", {})
        if structure_id not in structures:
            raise KeyError(f"未知结构：{structure_id}")
        structure_record = structures[structure_id]
        branch_id = structure_record.get("branch_id")
        branches = manager.data.get("branches", {})
        if branch_id not in branches:
            raise KeyError(f"结构 {structure_id} 引用了未知 branch：{branch_id}")
        stage = task["stage"]
        model = config.get("mlip") or {}
        return {
            "manager": manager,
            "structure_record": deepcopy(structure_record),
            "structure": structure_record.get("source_path") or task.get("structure_path"),
            "branch": deepcopy(branches[branch_id]),
            "boundary": deepcopy(manager.boundary),
            "phase_references": phase_references,
            "model_path": model.get("model_path"),
            "model_version": model.get("name"),
            "work_directory": Path(
                task.get("work_directory") or (root / structure_id / stage / task["task_id"])
            ),
            "mlip_backend": _backend(configured_backends, "mlip_backend", "relax_and_feature"),
            "mc_backend": _backend(configured_backends, "mc_backend", "deep_search"),
            "dft_singlepoint_backend": _backend(
                configured_backends, "dft_singlepoint_backend", "dft_single_point"
            ),
            "dft_relax_backend": _backend(configured_backends, "dft_relax_backend", "dft_relax"),
        }

    dispatcher = create_registry_dispatcher(stage_registry, context_factory)
    dispatcher.context_factory = context_factory
    dispatcher.registry = stage_registry
    return dispatcher


def _backend(backends: dict[str, Callable[..., dict[str, Any]]], explicit: str, stage: str):
    return backends.get(explicit, backends.get(stage))

"""Prepare and execute self-contained MLIP Relax/MC Slurm task payloads."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from scientific_layer.mlip.mace_worker import run_mace_worker


def create_mlip_task_preparer(manager, phase_references, config):
    """Capture ledger context and return a JSON-safe Slurm task preparer."""
    model = config.get("mlip") or {}

    def prepare(task):
        if task.get("stage") not in {"relax_and_feature", "deep_search"}:
            return deepcopy(task)
        current = deepcopy(task)
        structure_id = _resolve_structure_id(manager, current)
        record = manager.data["structures"][structure_id]
        structure_path = current.get("structure_path") or record.get("source_path")
        if not structure_path:
            raise ValueError(f"MLIP 任务 {current.get('task_id')} 缺少结构文件路径")
        branch = deepcopy(manager.data["branches"][record["branch_id"]])
        segment_budget = current.get("incremental_budget")
        if segment_budget is None:
            budget = current.get("budget")
            segment_budget = budget.get("steps") if isinstance(budget, dict) else budget
        model_paths = [str(path) for path in (model.get("model_paths") or [])]
        base_parameters = {
            "device": model.get("device", "cuda"),
            "mace_head": _resolve_mace_head(model),
            "main_model_index": int(model.get("main_model_index", 0)),
            "save_relax_traj": False,
        }
        if current["stage"] == "deep_search":
            full_na_path = (current.get("full_na_structure_path")
                            or record.get("full_na_structure_path")
                            or branch.get("full_na_structure_path"))
            if not full_na_path or not Path(full_na_path).is_file():
                full_na_path = _materialize_full_na_template(
                    manager, branch, record, phase_references, config
                )
            base_parameters["full_na_structure"] = str(full_na_path)
        current.update({
            "structure_id": structure_id,
            "object_id": structure_id,
            "branch_id": record["branch_id"],
            "worker_job": {
                "operation": "mc" if current["stage"] == "deep_search" else "relax",
                "structure_path": str(structure_path),
                "model_path": None if model_paths else str(model.get("model_path") or ""),
                "model_paths": model_paths,
                "model_version": model.get("version") or model.get("name"),
                "output_directory": current.get("calculation_directory"),
                "parameters": {**base_parameters,
                    **(deepcopy(model.get("relax_parameters") or {}) if current["stage"] == "relax_and_feature" else {}),
                    **(deepcopy(model.get('mc_parameters') or {}) if current['stage'] == 'deep_search' else {}),
                    **deepcopy(current.get("parameters") or {})},
                "segment_budget": int(segment_budget) if segment_budget is not None else None,
                "checkpoint": current.get("checkpoint"),
                "branch": branch,
                "boundary": deepcopy(manager.boundary),
                "phase_references": deepcopy(phase_references),
            },
        })
        return current

    return prepare


def _resolve_mace_head(model):
    configured = model.get("mace_head")
    if configured:
        return configured
    identity = " ".join(str(model.get(key) or "") for key in ("name", "version", "model_path")).lower()
    if "mace-mh-1" in identity or "mh-1" in identity or "mh_1" in identity:
        return "omat_pbe"
    return None


def _materialize_full_na_template(manager, branch, record, phase_references, config):
    from fractions import Fraction
    from scientific_layer.structures.boundary_utils import normalize_fraction
    from scientific_layer.structures.build_mc_full_na_template import build_mc_full_na_template

    if Fraction(normalize_fraction(branch["x"])) == 0:
        raise ValueError(f"branch {branch.get('branch_id')} 为 Na0，不需要也不能执行 Na/V MC")
    structure = build_mc_full_na_template(
        branch, manager.boundary, phase_references, config=config)
    source = Path(record.get("source_path") or "")
    if not source.parent.is_dir():
        raise ValueError(f"structure {record.get('structure_id')} 缺少可写的本地结构目录")
    path = source.parent / "full_na_structure.vasp"
    if not path.exists():
        structure.to(filename=path, fmt="poscar")
    branch["full_na_structure_path"] = str(path)
    record["full_na_structure_path"] = str(path)
    return path


def execute_mlip_task(task):
    """Executor reference for ``run_slurm_array_task --executor``."""
    job = deepcopy(task.get("worker_job") or {})
    if not job:
        raise ValueError("MLIP task 缺少 worker_job")
    job["output_directory"] = str(Path(task["result_path"]).parent)
    structure_path = Path(job.get("structure_path") or "")
    if not structure_path.is_absolute():
        job["structure_path"] = str(Path(task["result_path"]).parent / structure_path)
    checkpoint = job.get("checkpoint")
    if checkpoint and not Path(checkpoint).is_absolute():
        job["checkpoint"] = str(Path(task["result_path"]).parent / checkpoint)
    full_na = (job.get("parameters") or {}).get("full_na_structure")
    if full_na and not Path(full_na).is_absolute():
        job["parameters"]["full_na_structure"] = str(
            Path(task["result_path"]).parent / full_na
        )
    if not job.get("model_path") and not job.get("model_paths"):
        raise ValueError("MLIP task 缺少 model_path/model_paths")
    result = run_mace_worker(job)
    outputs = deepcopy(result)
    outputs.setdefault("mlip_name", "MACE")
    outputs.setdefault("mlip_version", job.get("model_version"))
    return {
        "status": result.get("status", "completed"),
        "stage": task["stage"],
        "structure_id": task["structure_id"],
        "converged": (result.get("relax_stopped_normally") is True
                      if task["stage"] in {"relax_and_feature", "deep_search"}
                      else result.get("converged")),
        "relax_stopped_normally": result.get("relax_stopped_normally"),
        "checkpoint": result.get("checkpoint"),
        "result_path": result.get("structure_path") or str(Path(task["result_path"]).parent),
        "outputs": outputs,
        "max_mc_steps": result.get("max_mc_steps"),
        "requested_max_mc_steps": result.get("requested_max_mc_steps"),
        "patience_steps": result.get("patience_steps"),
        "min_improvement": result.get("min_improvement"),
        "actual_mc_steps": result.get("actual_mc_steps"),
        "stop_reason": result.get("stop_reason"),
        "restart_mode": result.get("restart_mode"),
        "strict_chain_resume": result.get("strict_chain_resume"),
        "actual_gpu_core_hours": result.get("actual_gpu_core_hours"),
        "job_accounting": result.get("job_accounting"),
        "actual_cost": result.get("actual_cost"),
        "model_version": job.get("model_version"),
    }


def _resolve_structure_id(manager, task):
    structure_id = task.get("structure_id")
    if structure_id in manager.data.get("structures", {}):
        return structure_id
    branch_id = task.get("branch_id") or task.get("object_id")
    branch = manager.data.get("branches", {}).get(branch_id) or {}
    candidates = list(branch.get("structure_ids") or [])
    if not candidates:
        raise ValueError(f"Branch {branch_id!r} 没有可供 MLIP 计算的结构")
    return sorted(candidates)[0]

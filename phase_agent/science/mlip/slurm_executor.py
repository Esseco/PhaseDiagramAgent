"""Prepare and execute self-contained MLIP Relax/MC Slurm task payloads."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from phase_agent.science.mlip.mace_worker import run_mace_worker


def remote_model_references(model):
    """Serialize recorded HPC references verbatim; never inspect local weights."""
    members = model.get("members") or []
    replacements = {
        str(member["model_path"]): str(member.get("remote_model_path") or member["model_path"])
        for member in members
        if member.get("model_path")
    }
    declared = model.get("remote_model_paths") or model.get("model_paths") or []
    if not declared and members:
        declared = [m.get("remote_model_path") or m.get("model_path") for m in members]
    if not isinstance(declared, (list, tuple)):
        raise ValueError("remote_model_reference_invalid: committee paths must be a list")
    paths = [replacements.get(str(path), str(path)) if path else "" for path in declared]
    main = None if paths else str(model.get("remote_model_path") or model.get("model_path") or "")
    selected = paths or ([main] if main else [])
    if not selected or any(not name.strip() for name in selected):
        raise ValueError(
            "remote_model_reference_missing: configure HPC model paths; local weights are not required"
        )
    if len(selected) != len(set(selected)):
        raise ValueError("remote_model_reference_invalid: duplicate committee paths")
    if members and len(paths) != len(members):
        raise ValueError("remote_model_reference_invalid: committee member/path count mismatch")
    if paths and not 0 <= int(model.get("main_model_index", 0)) < len(paths):
        raise ValueError("remote_model_reference_invalid: main_model_index out of range")
    evidence = []
    for name in selected:
        row = next(
            (
                m
                for m in members
                if str(m.get("remote_model_path") or m.get("model_path") or "") == name
            ),
            model if name == main else {},
        )
        evidence.append(
            {"model_path": name, "sha256": row.get("sha256"), "size_bytes": row.get("size_bytes")}
        )
    return main, paths, evidence


def create_mlip_task_preparer(manager, phase_references, config):
    """Capture ledger context and return a JSON-safe Slurm task preparer."""
    model = config.get("mlip") or {}
    history_tasks = None

    def prepare(task):
        nonlocal history_tasks
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
        model_path, model_paths, model_evidence = remote_model_references(model)
        base_parameters = {
            "device": model.get("device", "cuda"),
            "mace_head": _resolve_mace_head(model),
            "main_model_index": int(model.get("main_model_index", 0)),
            "save_relax_traj": False,
        }
        if current["stage"] == "deep_search":
            full_na_path = (
                current.get("full_na_structure_path")
                or record.get("full_na_structure_path")
                or branch.get("full_na_structure_path")
            )
            if not full_na_path or not Path(full_na_path).is_file():
                full_na_path = _materialize_full_na_template(
                    manager, branch, record, phase_references, config
                )
            base_parameters["full_na_structure"] = str(full_na_path)
            from phase_agent.science.mc.project_occupation_history import history_seed

            if history_tasks is None:
                import json

                path = Path(config["state_path"]) if config.get("state_path") else None
                saved = (
                    json.loads(path.read_text(encoding="utf-8")) if path and path.is_file() else {}
                )
                history_tasks = [
                    t for t in saved.get("tasks", []) if t.get("stage") == "deep_search"
                ]

            base_parameters["occupation_history_scope"] = record["branch_id"]
            base_parameters["occupation_history_seed"] = history_seed(
                config, record["branch_id"], full_na_path, tasks=history_tasks
            )
            base_parameters["max_duplicate_trials"] = (config.get("mc_policy") or {}).get(
                "max_duplicate_trials", 100
            )
        current.update(
            {
                "structure_id": structure_id,
                "object_id": structure_id,
                "branch_id": record["branch_id"],
                "worker_job": {
                    "operation": (
                        "mc"
                        if current["stage"] == "deep_search"
                        else "predict"
                        if (current.get("parameters") or {}).get("model_refresh_operation")
                        == "predict"
                        else "relax"
                    ),
                    "structure_path": str(structure_path),
                    "model_path": model_path,
                    "model_paths": model_paths,
                    "model_file_evidence": model_evidence,
                    "model_storage": "remote",
                    "model_version": model.get("version") or model.get("name"),
                    "output_directory": current.get("calculation_directory"),
                    "parameters": {
                        **base_parameters,
                        **(
                            deepcopy(model.get("relax_parameters") or {})
                            if current["stage"] == "relax_and_feature"
                            else {}
                        ),
                        **(
                            deepcopy(model.get("mc_parameters") or {})
                            if current["stage"] == "deep_search"
                            else {}
                        ),
                        **deepcopy(current.get("parameters") or {}),
                    },
                    "segment_budget": int(segment_budget) if segment_budget is not None else None,
                    **(
                        {"stopping_threshold_source": current["stopping_threshold_source"]}
                        if current.get("stopping_threshold_source")
                        else {}
                    ),
                    "checkpoint": current.get("checkpoint"),
                    "branch": branch,
                    "boundary": deepcopy(manager.boundary),
                    "phase_references": deepcopy(phase_references),
                },
            }
        )
        if (
            current["stage"] == "deep_search"
            and current.get("stopping_threshold_source") == "library_default"
        ):
            current["worker_job"]["parameters"].pop("min_improvement", None)
        return current

    return prepare


def _resolve_mace_head(model):
    configured = model.get("mace_head")
    if configured:
        return configured
    identity = " ".join(
        str(model.get(key) or "") for key in ("name", "version", "model_path")
    ).lower()
    if "mace-mh-1" in identity or "mh-1" in identity or "mh_1" in identity:
        return "omat_pbe"
    return None


def _materialize_full_na_template(manager, branch, record, phase_references, config):
    from fractions import Fraction
    from phase_agent.science.structures.boundary_utils import normalize_fraction
    from phase_agent.science.structures.build_mc_full_na_template import (
        build_mc_full_na_template,
    )

    if Fraction(normalize_fraction(branch["x"])) == 0:
        raise ValueError(f"branch {branch.get('branch_id')} 为 Na0，不需要也不能执行 Na/V MC")
    structure = build_mc_full_na_template(branch, manager.boundary, phase_references, config=config)
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
        job["parameters"]["full_na_structure"] = str(Path(task["result_path"]).parent / full_na)
    if not job.get("model_path") and not job.get("model_paths"):
        raise ValueError("MLIP task 缺少 model_path/model_paths")
    root = Path(task["result_path"]).parent
    if job.get("model_paths"):
        job["model_paths"] = [
            str(Path(p) if Path(p).is_absolute() else root / p) for p in job["model_paths"]
        ]
    elif not Path(job["model_path"]).is_absolute():
        job["model_path"] = str(root / job["model_path"])
    result = run_mace_worker(job)
    outputs = deepcopy(result)
    outputs.setdefault("mlip_name", "MACE")
    outputs.setdefault("mlip_version", job.get("model_version"))
    response = {
        "status": result.get("status", "completed"),
        "stage": task["stage"],
        "structure_id": task["structure_id"],
        "converged": (
            result.get("optimizer_converged") is True
            if job["operation"] == "relax"
            else result.get("relax_stopped_normally") is True
            if job["operation"] == "mc"
            else result.get("converged")
        ),
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
    from phase_agent.analysis.convergence.mlip_relaxation import (
        relaxation_failure,
        reject_relaxation,
    )

    reason = relaxation_failure({**task, **response}, require_optimizer=job["operation"] == "relax")
    if reason:
        reject_relaxation(response, reason)
    return response


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

"""Prepare approved MACE Relax tasks from existing structures for manual upload."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from phase_agent.tools.budget.estimate_stage_cost import estimate_stage_cost
from phase_agent.tools.budget.reserve_budget import reserve_budget
from phase_agent.tools.remote.manual_upload_runner import ManualUploadBatchRunner
from phase_agent.science.mlip.slurm_executor import create_mlip_task_preparer
from phase_agent.configuration.defaults.default_slurm_cluster_config import (
    default_slurm_cluster_config,
)


def prepare_relax_upload_batches(*, action, context):
    config = context["effective_config"]
    manager = context["manager"]
    state = deepcopy(context.get("event_state") or {})
    if (state.get("dedup_gate") or {}).get("status") != "ready":
        return {"status": "not_configured", "reason": "structure_dedup_not_ready", "state": state}
    model = config.get("mlip") or {}
    version = model.get("version") or model.get("name")
    if not version or not (model.get("model_path") or model.get("model_paths")):
        return {"status": "not_configured", "reason": "remote_mlip_model_missing", "state": state}
    root = config.get("upload_batches_directory")
    command = ((config.get("supercomputer") or {}).get("worker") or {}).get("command") or []
    if not root or not command or any("YOUR_" in str(value) for value in command):
        return {
            "status": "not_configured",
            "reason": "upload_directory_or_worker_command_missing",
            "state": state,
        }
    parameters = action.get("parameters") or {}
    if parameters.get("rebuild_inputs"):
        from phase_agent.tools.local.rebuild_relax_inputs import rebuild_relax_inputs

        state = rebuild_relax_inputs(state, root, parameters.get("cleanup_plan"))
    requested = list(
        dict.fromkeys(
            (action.get("target_ids") or [])
            if parameters.get("selection_scope") == "target_ids"
            else manager.data.get("branches", {})
        )
    )
    unknown = [bid for bid in requested if bid not in manager.data.get("branches", {})]
    if unknown:
        return {
            "status": "rejected",
            "reason": "unknown_branch_ids",
            "branch_ids": unknown,
            "state": state,
        }
    count = min(3, int((config.get("bohb") or {}).get("relax_structures_per_branch", 3)))
    valid_ids = set((state.get("dedup_gate") or {}).get("valid_structure_ids") or [])
    planned = []
    for bid in requested:
        branch = manager.data["branches"][bid]
        ids = [
            sid
            for sid in branch.get("structure_ids") or []
            if sid in manager.data["structures"]
            and (not valid_ids or sid in valid_ids)
            and (manager.data["structures"][sid].get("metadata") or {}).get("initialization_method")
            == "electrostatic_top10_random3_layer_occupied"
        ][:count]
        for sid in ids:
            record = manager.data["structures"][sid]
            path = Path(record.get("source_path") or "")
            if not path.is_file():
                return {
                    "status": "not_configured",
                    "reason": f"structure_file_missing:{sid}",
                    "state": state,
                }
            atoms = sum((record.get("composition") or {}).values()) or (
                record.get("metadata") or {}
            ).get("atom_count")
            if not atoms:
                from pymatgen.core import Structure

                atoms = len(Structure.from_file(path))
            cost = estimate_stage_cost(
                "relax_and_feature", atom_count=int(atoms), budgets=config["budgets"]
            )["value"]
            planned.append((bid, sid, cost))
    if not planned:
        return {
            "status": "not_configured",
            "reason": "no_legal_existing_relax_structures",
            "state": state,
        }
    settings = deepcopy(model.get("relax_parameters") or {})
    settings_id = hashlib.sha256(
        json.dumps(settings, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]
    from phase_agent.decisions.agent.choose_debug_next_action import _verified_migrated_relax_ids

    migrated_ids = _verified_migrated_relax_ids(state, model, version)
    created = []
    for bid, sid, cost in planned:
        key = f"relax-screen:{version}:{settings_id}:{sid}"
        existing = next(
            (row for row in state.get("tasks") or [] if row.get("task_key") == key), None
        )
        if existing is None and sid in migrated_ids:
            continue
        if existing:
            if existing.get("status") == "completed" or _task_files_complete(existing):
                continue
            for field in (
                "slurm_batch_id",
                "slurm_array_index",
                "batch_id",
                "input_path",
                "result_path",
                "task_checksum",
            ):
                existing.pop(field, None)
            existing["status"] = "pending"
            if existing not in state.setdefault("pending_tasks", []):
                state["pending_tasks"].append(existing)
            continue
        reservation = reserve_budget(
            state,
            task_key=key,
            stage="relax_and_feature",
            amount=cost,
            limits=config["budgets"],
            config_version=context["config_version"],
            model_version=version,
        )
        if reservation["status"] != "reserved":
            return {
                "status": "budget_exhausted",
                "reason": reservation.get("reasons"),
                "state": context.get("event_state") or {},
            }
        state = reservation["state"]
        task = {
            "task_id": "RELAX-" + hashlib.sha256(key.encode()).hexdigest()[:12],
            "task_key": key,
            "branch_id": bid,
            "structure_id": sid,
            "object_id": sid,
            "stage": "relax_and_feature",
            "status": "pending",
            "model_version": version,
            "config_version": context["config_version"],
            "planned_relative_cost": cost,
            "parameters": deepcopy(settings),
            "screening_basis": "electrostatic_top10_random3_layer_occupied",
        }
        task["generation_cycle"] = len(state.get("generation_history") or [])
        task["parent_decision_id"] = context.get("approval_record_id")
        state.setdefault("tasks", []).append(task)
        state.setdefault("pending_tasks", []).append(task)
        created.append(task)
    runner = ManualUploadBatchRunner(
        root,
        worker_command=command,
        stage_batch_sizes=((config.get("supercomputer") or {}).get("batch_sizes") or {}),
        stage_profiles=default_slurm_cluster_config(),
        task_preparer=create_mlip_task_preparer(
            manager, context.get("phase_references") or {}, config
        ),
    )
    batches = []
    while True:
        prepared = runner.prepare(state)
        state = prepared["state"]
        if prepared["status"] == "no_tasks":
            break
        batches.append(prepared["batch"])
    from phase_agent.tools.remote.write_relax_upload_plan import write_relax_upload_plan

    pending_relax = any(
        row.get("stage") == "relax_and_feature"
        and row.get("status") in {"pending", "running"}
        and row.get("input_path")
        for row in state.get("tasks") or []
    )
    upload_plan = write_relax_upload_plan(state, root) if pending_relax else None
    return {
        "status": "prepared" if batches else "already_prepared",
        "state": state,
        "task_count": len(created),
        "batch_count": len(batches),
        "upload_plan_path": str(upload_plan) if upload_plan else None,
        "batches": [
            {
                "batch_id": row["batch_id"],
                "directory": row["upload_directory"],
                "results_directory": row["results_directory"],
                "task_directory": row.get("task_directory"),
                "task_directories": row.get("task_directories"),
                "task_ids": row["task_ids"],
            }
            for row in batches
        ],
        "submitted": False,
    }


def _task_files_complete(task):
    input_path = Path(task.get("input_path") or "")
    if not input_path.is_file():
        return False
    directory = input_path.parent
    return (
        all((directory / name).is_file() for name in ("initial.vasp", "task.json"))
        and (
            (directory.parent / "run_mlip_batch.py").is_file()
            or (directory / "run_mlip_task.py").is_file()
        )
        and (directory.parent / "GPU.sh").is_file()
    )

"""Prepare approved MACE Relax tasks from existing structures for manual upload."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import zipfile

from phase_agent.tools.budget.estimate_stage_cost import estimate_stage_cost
from phase_agent.tools.budget.reserve_budget import reserve_owned_budget
from phase_agent.tools.remote.manual_upload_runner import ManualUploadBatchRunner
from phase_agent.science.mlip.slurm_executor import create_mlip_task_preparer
from phase_agent.configuration.defaults.default_slurm_cluster_config import (
    default_slurm_cluster_config,
)


def prepare_relax_upload_batches(*, action, context):
    config = context["effective_config"]
    manager = context["manager"]
    original = context.get("event_state") or {}
    exact = (action.get("parameters") or {}).get("selection_scope") == "candidate_ids"
    if exact:
        # Only these collections are mutated by budget reservation and packaging.
        # Copy them together to preserve shared task references and rollback on failure.
        state = dict(original)
        state.update(deepcopy({k: original[k] for k in
                               ("tasks", "pending_tasks", "slurm_batches", "budget_reservations")
                               if k in original}))
    else:
        state = deepcopy(original)
    if (state.get("dedup_gate") or {}).get("status") != "ready":
        return {"status": "not_configured", "reason": "structure_dedup_not_ready", "state": state}
    model = config.get("mlip") or {}
    if (action.get("parameters") or {}).get("selection_scope") == "candidate_ids":
        active = state.get("active_model") or {}
        if active.get("version") != state.get("active_model_version"):
            return {"status": "rejected", "reason": "active_model_identity_mismatch", "state": state}
        config = deepcopy(config)
        model = config["mlip"] = {**model, **active}
    version = model.get("version") or model.get("name")
    from phase_agent.science.mlip.slurm_executor import remote_model_references

    if not version:
        return {"status": "not_configured", "reason": "remote_mlip_model_missing", "state": state}
    try:
        remote_model_references(model)
    except ValueError as error:
        return {"status": "not_configured", "reason": str(error), "state": state}
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
    exact = parameters.get("selection_scope") == "candidate_ids"
    if exact:
        try:
            planned = _exact_candidate_frames(action, state, manager, config, version)
        except (ValueError, OSError, KeyError) as error:
            return {"status": "rejected", "reason": str(error), "state": state}
    requested = (
        []
        if exact
        else list(
            dict.fromkeys(
                (action.get("target_ids") or [])
                if parameters.get("selection_scope") == "target_ids"
                else manager.data.get("branches", {})
            )
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
    if not exact:
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
            planned.append((bid, sid, cost, None))
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
    from phase_agent.agents.strategic.choose_debug_next_action import _verified_migrated_relax_ids

    migrated_ids = _verified_migrated_relax_ids(state, model, version)
    created = []
    approved_keys = set()
    for bid, sid, cost, frame in planned:
        key = (
            f"relax-frame:{version}:{context['config_version']}:{settings_id}:{sid}:{frame['sha256']}"
            if frame
            else f"relax-screen:{version}:{settings_id}:{sid}"
        )
        approved_keys.add(key)
        existing = next(
            (row for row in state.get("tasks") or [] if row.get("task_key") == key), None
        )
        if not exact and existing is None and sid in migrated_ids:
            continue
        if existing:
            if existing.get("status") == "completed" or _task_files_complete(existing):
                continue
            bound = any(
                existing.get(k) for k in ("slurm_batch_id", "batch_id", "input_path", "job_id")
            )
            if bound and not existing.get("recovery_cancelled"):
                try:
                    _restore_registered_inputs(state, existing, root)
                except (OSError, ValueError, KeyError, zipfile.BadZipFile, EOFError) as error:
                    return {
                        "status": "not_configured",
                        "reason": "原输入恢复校验未通过：" + str(error),
                        "task_ids": [existing["task_id"]],
                        "recovery": "保留原批次、作业和回执；补回原上传包或修复指定文件后重试，不重投计算。",
                        "state": context.get("event_state") or {},
                    }
                if _task_files_complete(existing):
                    continue
            if bound or existing.get("status") != "pending" or existing.get("recovery_cancelled"):
                return {
                    "status": "not_configured",
                    "reason": "existing_relax_inputs_require_recovery",
                    "task_ids": [existing["task_id"]],
                    "recovery": "原输入缺失或已有执行/取消记录；从原上传归档恢复，执行状态不明时走审批页恢复核对。只有明确审核过的未提交输入重建可解除原绑定。",
                    "state": context.get("event_state") or {},
                }
            if existing not in state.setdefault("pending_tasks", []):
                state["pending_tasks"].append(existing)
            continue
        reservation = reserve_owned_budget(
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
        if frame:
            task.update(
                structure_path=frame["path"],
                input_structure_sha256=frame["sha256"],
                candidate_id=frame["candidate_id"],
                screening_basis="exact_candidate_frame",
            )
        task["generation_cycle"] = len(state.get("generation_history") or [])
        task["parent_decision_id"] = context.get("approval_record_id")
        state.setdefault("tasks", []).append(task)
        state.setdefault("pending_tasks", []).append(task)
        created.append(task)
    runner = ManualUploadBatchRunner(
        root,
        task_ids={
            row["task_id"]
            for row in state.get("tasks") or []
            if row.get("task_key") in approved_keys
        },
        worker_command=command,
        stage_batch_sizes=((config.get("supercomputer") or {}).get("batch_sizes") or {}),
        stage_profiles=default_slurm_cluster_config(),
        task_preparer=create_mlip_task_preparer(
            manager, context.get("phase_references") or {}, config
        ),
    )
    batches = []
    while True:
        prepared = runner.prepare(state, _owned=exact)
        state = prepared["state"]
        if prepared["status"] == "no_tasks":
            break
        batches.append(prepared["batch"])
    if exact and batches:
        from phase_agent.tools.remote.prepare_phase_analysis import prepare_phase_analysis

        prepared_ids = {task["task_id"] for task in created}
        batch_ids = {tid for batch in batches for tid in batch.get("task_ids") or []}
        if prepared_ids != batch_ids:
            raise ValueError("exact_relax_prepared_task_scope_mismatch")
        prepare_phase_analysis(
            state, batches, manager, context.get("phase_references"), default_slurm_cluster_config()
        )
    from phase_agent.tools.remote.write_relax_upload_plan import write_relax_upload_plan

    pending_relax = any(
        row.get("stage") == "relax_and_feature"
        and row.get("status") in {"pending", "running"}
        and row.get("input_path")
        for row in state.get("tasks") or []
    )
    upload_plan = write_relax_upload_plan(state, root) if pending_relax else None
    from phase_agent.tools.remote.submission_handoff import finalize_prepared_handoffs

    delivery = finalize_prepared_handoffs(state)
    if delivery.get("status") == "not_configured":
        return delivery
    return {
        "status": "prepared" if batches else "already_prepared",
        **delivery,
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


def _exact_candidate_frames(action, state, manager, config, version):
    """Resolve approved candidate frames without branch resampling or path substitution."""
    params = action.get("parameters") or {}
    ids = action.get("target_ids") or []
    hashes = params.get("candidate_frame_sha256") or {}
    if not ids or len(ids) != len(set(ids)) or set(ids) != set(hashes):
        raise ValueError("candidate_frame_identity_invalid")
    if state.get("active_model_version") != version or params.get("model_version") != version:
        raise ValueError("candidate_frame_model_changed")
    candidates = {}
    for row in state.get("qbc_candidates") or []:
        if row.get("candidate_id") in candidates:
            raise ValueError("candidate_frame_id_ambiguous")
        candidates[row.get("candidate_id")] = row
    result = []
    for cid in ids:
        row = candidates.get(cid) or {}
        sid = row.get("structure_id")
        record = manager.data.get("structures", {}).get(sid)
        if not record:
            raise ValueError("candidate_structure_unregistered:" + str(cid))
        bid = record.get("branch_id")
        if bid not in manager.data.get("branches", {}):
            raise ValueError("candidate_branch_unregistered:" + str(cid))
        path = Path(row.get("structure_path") or "")
        digest = hashes[cid]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or row.get("structure_sha256") != digest
            or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise ValueError("candidate_frame_checksum_mismatch:" + str(cid))
        atoms = row.get("atom_count") or sum((record.get("composition") or {}).values())
        if not atoms or int(atoms) != atoms:
            raise ValueError("candidate_atom_count_missing:" + str(cid))
        cost = estimate_stage_cost(
            "relax_and_feature", atom_count=int(atoms), budgets=config["budgets"]
        )["value"]
        result.append((bid, sid, cost, {"candidate_id": cid, "path": str(path), "sha256": digest}))
    return result


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


def _restore_registered_inputs(state, task, root):
    """Restore immutable original inputs only; never reset a submitted task identity."""
    from phase_agent.tools.remote.submission_storage import restore_storage_directory

    batch_id = task.get("slurm_batch_id") or task.get("batch_id")
    batch = next(
        (row for row in state.get("slurm_batches") or [] if row.get("batch_id") == batch_id), {}
    )
    if not batch.get("upload_directory") or not task.get("input_path"):
        return
    directory = Path(batch["upload_directory"]).resolve()
    source = Path(task["input_path"]).resolve()
    if not directory.is_relative_to(Path(root).resolve()) or source.parent.parent != directory:
        raise ValueError("registered_input_path_outside_upload_batch")
    restore_storage_directory(directory.parent, "inputs")

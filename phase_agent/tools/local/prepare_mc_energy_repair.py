"""Approved, same-frame primary-model correction for legacy MC mean energies."""

from copy import deepcopy
import hashlib
import math
from pathlib import Path

from phase_agent.tools.remote.integrity import payload_checksum, file_checksum
from phase_agent.analysis.phase.energy_consistency import legacy_mc_mean_record_ids
from phase_agent.tools.budget.estimate_stage_cost import estimate_stage_cost


def _primary_model(state, config):
    active = state.get("active_model") or {}
    version = state.get("active_model_version")
    if not version or active.get("version") != version:
        raise ValueError("energy_repair_active_model_mismatch")
    cfg = config.get("mlip") or {}
    index = cfg.get("main_model_index", active.get("main_model_index", 0))
    if active.get("main_model_index") is not None and active["main_model_index"] != index:
        raise ValueError("energy_repair_main_index_mismatch")
    if type(index) is not int or index < 0:
        raise ValueError("energy_repair_main_index_invalid")
    members = active.get("members") or []
    if members:
        if index >= len(members):
            raise ValueError("energy_repair_main_member_missing")
        member = members[index]
        path = member.get("remote_model_path") or member.get("model_path")
        digest = member.get("sha256")
        if member.get("is_main_model") is not True:
            raise ValueError("energy_repair_main_member_unverified")
        if active.get("sha256") and active["sha256"] != digest:
            raise ValueError("energy_repair_main_checksum_mismatch")
        active_path = active.get("remote_model_path") or active.get("model_path")
        if active_path and active_path != path:
            raise ValueError("energy_repair_main_path_mismatch")
    else:
        path = active.get("remote_model_path") or active.get("model_path")
        digest = active.get("sha256")
    if (
        not isinstance(path, str)
        or not path.startswith("/")
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("energy_repair_main_model_evidence_missing")
    return {
        "version": version,
        "path": path,
        "sha256": digest,
        "source_index": index,
        "framework": cfg.get("framework", "mace"),
        "device": cfg.get("device", "cuda"),
        "mace_head": cfg.get("mace_head"),
    }


def preview(action, state, manager, config, config_version):
    ids = action.get("target_ids") or []
    if (
        not isinstance(ids, list)
        or not ids
        or any(not isinstance(t, str) or not t for t in ids)
        or len(ids) != len(set(ids))
    ):
        raise ValueError("energy_repair_targets_invalid")
    if config_version != state.get("confirmed_config_version"):
        raise ValueError("energy_repair_config_changed")
    model = _primary_model(state, config)
    hull = (state.get("phase_diagrams") or {}).get("mlip") or {}
    if (
        hull.get("status") != "completed"
        or hull.get("model_version") != model["version"]
        or not hull.get("version")
    ):
        raise ValueError("energy_repair_hull_unavailable")
    suspect = set(legacy_mc_mean_record_ids(state, hull))
    if not suspect:
        raise ValueError("energy_repair_no_proven_mixed_hull")
    records = {}
    for record in state.get("phase_records") or []:
        tid = record.get("source_task_id")
        if tid in ids:
            if tid in records:
                raise ValueError("energy_repair_duplicate_source_record")
            records[tid] = record
    tasks = {}
    for task in state.get("tasks") or []:
        tid = task.get("task_id")
        if tid in ids:
            if tid in tasks:
                raise ValueError("energy_repair_duplicate_task_id")
            tasks[tid] = task
    if set(tasks) != set(ids) or set(records) != set(ids):
        raise ValueError("energy_repair_source_scope_missing")
    if not suspect.issubset({row.get("record_id") for row in records.values()}):
        raise ValueError("energy_repair_mixed_hull_scope_incomplete")
    frames = []
    cross_model = False
    for tid in sorted(ids):
        task = tasks[tid]
        out = task.get("outputs") or {}
        record = records[tid]
        source_model = task.get("model_version")
        if not source_model or record.get("model_version") != source_model:
            raise ValueError("energy_repair_source_model_binding:" + tid)
        if source_model != model["version"]:
            rounds = (state.get("upload_layout") or {}).get("model_rounds") or {}
            source_round, target_round = rounds.get(source_model), rounds.get(model["version"])
            if (type(source_round) is not int or type(target_round) is not int
                    or source_round < 1 or source_round >= target_round):
                raise ValueError("energy_repair_source_model_order:" + tid)
            cross_model = True
        qbc = out.get("qbc") or {}
        pool = out.get("pool_summary") or []
        mean = pool[0].get("energy_mean_per_atom") if pool and isinstance(pool[0], dict) else None
        energy = out.get("energy_per_atom")
        if (
            task.get("stage") != "deep_search"
            or task.get("status") != "completed"
            or task.get("checks_passed", True) is not True
            or type(qbc.get("member_count")) is not int
            or qbc["member_count"] < 2
            or out.get("energy_source") == "main_model_selected_frame"
            or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                for v in (mean, energy)
            )
            or not math.isclose(mean, energy, rel_tol=1e-10, abs_tol=1e-10)
        ):
            raise ValueError("energy_repair_source_not_verified:" + tid)
        sid = task.get("structure_id")
        branch = task.get("branch_id")
        structure = (manager.data.get("structures") or {}).get(sid) or {}
        path = Path(out.get("structure_path") or "")
        if (
            not sid
            or not branch
            or structure.get("branch_id") != branch
            or not path.is_file()
            or file_checksum(path) != out.get("structure_checksum")
            or record.get("structure_sha256") != out.get("structure_checksum")
            or record.get("structure_id") != sid
            or not isinstance(out.get("energy"), (int, float))
            or isinstance(out.get("energy"), bool)
            or not isinstance(record.get("energy"), (int, float))
            or isinstance(record.get("energy"), bool)
            or not math.isclose(out["energy"], record["energy"], rel_tol=1e-10, abs_tol=1e-10)
        ):
            raise ValueError("energy_repair_frame_binding_invalid:" + tid)
        atom_count = out.get("atom_count")
        if type(atom_count) is not int or atom_count <= 0:
            raise ValueError("energy_repair_atom_count_invalid:" + tid)
        if not math.isclose(out["energy"], energy * atom_count, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("energy_repair_energy_inconsistent:" + tid)
        proxy = estimate_stage_cost(
            "relax_and_feature", atom_count=atom_count, budgets=config["budgets"]
        )["value"]
        frames.append(
            {
                "source_task_id": tid,
                "source_model_version": source_model,
                "record_id": record["record_id"],
                "structure_id": sid,
                "branch_id": branch,
                "structure_path": str(path),
                "structure_sha256": out["structure_checksum"],
                "atom_count": atom_count,
                "phase_status": record.get("status"),
                "relative_cost": proxy,
            }
        )
    if not cross_model:
        for frame in frames:
            frame.pop("source_model_version")
    body = {
        "schema": 2 if cross_model else 1,
        "kind": "mc_main_model_same_frame",
        "config_version": config_version,
        "model": model,
        "hull_version": hull["version"],
        "frames": frames,
        "affected_hull_records": sorted(suspect),
        "task_count": len(frames),
        "estimated_relative_cost_upper": sum(x["relative_cost"] for x in frames),
        "cost_basis": "relax_upper_bound_for_single_point",
        "relaxation": False,
        "submitted": False,
    }
    return {**body, "checksum": payload_checksum(body)}


def attach_preview(action, state, context, config):
    if (action.get("parameters") or {}).get("mode") != "mc_energy_repair_inputs":
        return action
    result = deepcopy(action)
    plan = preview(
        result,
        state,
        context["manager"],
        config,
        context.get("config_version") or state.get("confirmed_config_version"),
    )
    existing = (action.get("parameters") or {}).get("energy_repair_preview")
    if existing is not None and existing != plan:
        raise ValueError("energy_repair_preview_changed")
    result.setdefault("parameters", {})["energy_repair_preview"] = plan
    result["budget"] = plan["estimated_relative_cost_upper"]
    result["expected_purpose"] = (
        f"对 {len(plan['frames'])} 个 MC 末态使用原主模型做同帧单点，统一相图能量口径。"
    )
    return result


def prepare(*, action, context):
    state = deepcopy(context["event_state"])
    config = deepcopy(context["effective_config"])
    approval = context.get("approval_record_id")
    frozen = (action.get("parameters") or {}).get("energy_repair_preview")
    if not approval or not isinstance(frozen, dict):
        raise ValueError("energy_repair_approval_or_preview_missing")
    current = preview(action, state, context["manager"], config, context.get("config_version"))
    if current != frozen:
        raise ValueError("energy_repair_preview_changed")
    repair_id = current["checksum"]
    if any(t.get("energy_basis_repair_id") == repair_id for t in state.get("tasks") or []):
        raise ValueError("energy_repair_already_registered")
    root = config.get("upload_batches_directory")
    command = ((config.get("supercomputer") or {}).get("worker") or {}).get("command") or []
    if not root or not command:
        raise ValueError("energy_repair_upload_not_configured")
    from phase_agent.tools.budget.reserve_budget import reserve_owned_budget

    tasks = []
    for row in current["frames"]:
        key = "mc-energy-repair:" + repair_id + ":" + row["source_task_id"]
        if any(t.get("task_key") == key for t in state.get("tasks") or []):
            raise ValueError("energy_repair_task_exists")
        reserved = reserve_owned_budget(
            state,
            task_key=key,
            stage="relax_and_feature",
            amount=row["relative_cost"],
            limits=config["budgets"],
            config_version=current["config_version"],
            model_version=current["model"]["version"],
        )
        if reserved["status"] != "reserved":
            raise ValueError("energy_repair_reservation_rejected:" + str(reserved.get("reasons")))
        task = {
            "task_id": "ENERGY-" + hashlib.sha256(key.encode()).hexdigest()[:16],
            "task_key": key,
            "stage": "relax_and_feature",
            "status": "pending",
            "structure_id": row["structure_id"],
            "object_id": row["structure_id"],
            "branch_id": row["branch_id"],
            "structure_path": row["structure_path"],
            "input_structure_sha256": row["structure_sha256"],
            "model_version": current["model"]["version"],
            "model_sha256": current["model"]["sha256"],
            "config_version": current["config_version"],
            "planned_relative_cost": row["relative_cost"],
            "parent_decision_id": approval,
            "energy_basis_repair_id": repair_id,
            "energy_repair_source_task_id": row["source_task_id"],
            "energy_repair_source_record_id": row["record_id"],
            "upload_operation_id": "mc-energy-repair-" + repair_id.split(":")[-1][:12],
            "parameters": {
                "model_refresh_operation": "predict",
                "energy_basis_repair": True,
                "main_model_index": 0,
            },
        }
        if current["schema"] == 2:
            task["energy_repair_source_model_version"] = row["source_model_version"]
        state.setdefault("tasks", []).append(task)
        state.setdefault("pending_tasks", []).append(deepcopy(task))
        tasks.append(task)
    model = current["model"]
    config["mlip"] = {
        **(config.get("mlip") or {}),
        "version": model["version"],
        "model_path": model["path"],
        "remote_model_path": model["path"],
        "sha256": model["sha256"],
        "members": [],
        "model_paths": [],
        "remote_model_paths": [],
        "main_model_index": 0,
    }
    from phase_agent.tools.remote.manual_upload_runner import ManualUploadBatchRunner
    from phase_agent.science.mlip.slurm_executor import create_mlip_task_preparer
    from phase_agent.configuration.defaults.default_slurm_cluster_config import (
        default_slurm_cluster_config,
    )
    from phase_agent.tools.remote.prepare_options import supported_prepare_options
    from phase_agent.tools.remote.submission_handoff import finalize_prepared_handoffs

    base = create_mlip_task_preparer(
        context["manager"], context.get("phase_references") or {}, config
    )

    def portable(task):
        result = base(task)
        result.pop("structure_path", None)
        return result

    runner = ManualUploadBatchRunner(
        root,
        task_ids={t["task_id"] for t in tasks},
        worker_command=command,
        stage_batch_sizes=((config.get("supercomputer") or {}).get("batch_sizes") or {}),
        stage_profiles=default_slurm_cluster_config(),
        task_preparer=portable,
    )
    options = supported_prepare_options(
        runner.prepare, _owned=True, _defer_index=True, _defer_archive=True
    )
    batches = []
    while True:
        prepared = runner.prepare(state, **options)
        state = prepared["state"]
        if prepared["status"] == "no_tasks":
            break
        batches.append(prepared["batch"])
    delivery = finalize_prepared_handoffs(state)
    if delivery.get("status") == "not_configured":
        return delivery
    if sum(len(b["task_ids"]) for b in batches) != len(tasks):
        raise ValueError("energy_repair_packaging_incomplete")
    state.setdefault("mc_energy_repairs", {})[repair_id] = {
        "status": "prepared",
        "preview": current,
        "task_ids": [t["task_id"] for t in tasks],
        "approval_record_id": approval,
    }
    return {
        "status": "prepared",
        **delivery,
        "state": state,
        "task_count": len(tasks),
        "batch_count": len(batches),
        "submitted": False,
    }

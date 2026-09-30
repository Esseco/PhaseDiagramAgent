"""Rebuild the current approved MC upload inputs without submitting jobs."""

from copy import deepcopy
from collections import Counter
import hashlib
import math
from pathlib import Path
import shutil

from execution_layer.local.prepare_mc_upload_batches import prepare_mc_upload_batches
from execution_layer.mc_batch_policy import mc_batch_limit
from execution_layer.remote.batch_runner import _compatibility


def is_mc_regeneration_request(message):
    text = str(message or "").lower().replace(" ", "")
    return any(word in text for word in ("重新生成", "重新跑", "重跑", "重建")) and any(
        word in text for word in ("mc", "蒙特卡洛"))


def is_mc_regeneration_confirmation(message):
    return str(message or "").strip().lower().replace(" ", "") in {
        "确认重新生成当前轮mc任务", "确认重建当前轮mc任务"}


def plan_mc_regeneration(state, upload_root):
    """Validate the approved plan and exact current-round MC directory."""
    records = [row for row in state.get("action_records") or []
               if row.get("status") == "completed"
               and (row.get("final_action") or {}).get("tool") == "allocate_mc_bohb"
               and (row.get("human_feedback") or {}).get("decision") == "approve"]
    if not records:
        raise ValueError("没有已批准的当前轮 MC 分配方案")
    action = records[-1]["final_action"]
    preview = (action.get("parameters") or {}).get("budget_preview") or {}
    allocations = preview.get("allocations") or []
    if not allocations or len(allocations) != preview.get("selected_branch_count"):
        raise ValueError("已批准的 MC 分配清单不完整")
    if preview.get("estimated_relative_cost") is None:
        raise ValueError("已批准方案缺少 MC 成本估计")
    keys = [row.get("task_key") for row in allocations]
    if len(set(keys)) != len(keys) or not all(keys):
        raise ValueError("已批准的 MC 分配清单含重复或空任务键")
    current = [row for row in state.get("tasks") or []
               if row.get("stage") == "deep_search" and row.get("task_key") in set(keys)]
    if any(row.get("status") not in {"pending"} or row.get("job_id") for row in current):
        raise ValueError("当前轮 MC 已提交、完成或状态不明，不能重生成")
    old_ids = {row.get("task_id") for row in current}
    root = Path(upload_root).resolve()
    paths = {Path(row["input_path"]).resolve().parent.parent.parent for row in current
             if row.get("input_path")}
    if len(paths) != 1:
        raise ValueError("无法唯一确定当前轮 MC-search 目录")
    mc_root = paths.pop()
    if (mc_root.name != "MC-search" and not (
            mc_root.name.startswith("allocation-") and mc_root.parent.name == "MC-search")) or root not in mc_root.parents:
        if not ((mc_root.name.startswith('MC-round-') or '_MC-round-' in mc_root.name)
                and mc_root.parent.name.startswith('Search-group-') and root in mc_root.parents):
            raise ValueError("MC-search 目录不在配置的上传根目录内")
    other_tasks = [row for row in state.get("tasks") or []
                   if row.get("task_id") not in old_ids and row.get("input_path")
                   and mc_root in Path(row["input_path"]).resolve().parents]
    if other_tasks:
        raise ValueError("此目录还包含其他 MC 分配，请先拆分目录；不能删除历史输入或结果")
    batches = [row for row in state.get("slurm_batches") or []
               if (row.get("calculation_group") == "MC-search"
                   and Path(row.get("upload_directory") or "").resolve().parent == mc_root)
               or set(row.get("task_ids") or []) & old_ids]
    if any(row.get("job_id") or row.get("status") not in {"prepared", "superseded_inputs"}
           for row in batches):
        raise ValueError("MC 批次可能已提交，不能重生成")
    non_mc_batches = [row for row in state.get("slurm_batches") or [] if row not in batches]
    kept_ids = {row.get("batch_id") for row in non_mc_batches}
    remote_start = len(non_mc_batches) + 1
    if any(f"remote-{number:06d}" in kept_ids
           for number in range(remote_start, remote_start + len(allocations))):
        raise ValueError("保留的其他阶段批次编号与重建后的 MC 批次冲突")
    if any(Path(row.get("result_path") or "").is_file() for row in current):
        raise ValueError("发现 MC 结果文件，不能覆盖")
    if mc_root.exists():
        allowed = {"batch.snapshot.json", "full_na_structure.vasp", "GPU.sh",
                   "initial.vasp", "manifest.json", "run_mlip_batch.py",
                   "run_mlip_task.py", "SHA256SUMS", "task.json",
                   "UPLOAD_AND_SUBMIT.md"}
        for path in mc_root.rglob("*"):
            if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
                raise ValueError("MC-search 中含链接，不能自动删除")
            if path.is_file() and path.name not in allowed:
                raise ValueError(f"MC-search 中含结果、日志或未知文件，不能自动删除：{path}")
    branch_phase = {row.get("branch_id"): row.get("P")
                    for row in state.get("branch_candidates") or []}
    strata = {}
    for row in allocations:
        label = f"{row.get('tier') or 'unknown'}/{branch_phase.get(row.get('branch_id')) or 'unknown'}"
        strata[label] = strata.get(label, 0) + 1
    configured_batch_size = (((state.get("confirmed_config") or {}).get("supercomputer") or {})
                             .get("batch_sizes") or {}).get("deep_search", 10)
    batch_size = mc_batch_limit(configured_batch_size)
    compatibility_counts = Counter(_compatibility(row) for row in allocations)
    estimated_batches = sum(math.ceil(count / batch_size)
                            for count in compatibility_counts.values())
    return {"directory": str(mc_root), "exists": mc_root.exists(),
            "old_task_ids": sorted(old_ids), "old_batch_ids": sorted(
                row.get("batch_id") for row in batches if row.get("batch_id")),
            "old_batch_count": len(batches), "remote_batch_start": remote_start,
            "mc_sampling_start": 1, "allocation_count": len(allocations),
            "mc_batch_size": batch_size, "estimated_batch_count": estimated_batches,
            "allocation_by_tier_phase": dict(sorted(strata.items())),
            "total_mc_steps": sum(int(row["max_mc_steps"]) for row in allocations),
            "allocation_checksum": preview.get("allocation_checksum"),
            "estimated_relative_cost": preview.get("estimated_relative_cost")}


def regenerate_mc_inputs(state, *, approved_plan, upload_root, config, manager,
                         phase_references, config_version, allow_delete=False):
    plan = plan_mc_regeneration(state, upload_root)
    if plan != approved_plan:
        raise ValueError("MC 状态或目录已变化，请重新检查并确认")
    preview = next(row["final_action"]["parameters"]["budget_preview"]
                   for row in reversed(state["action_records"])
                   if row.get("status") == "completed"
                   and (row.get("final_action") or {}).get("tool") == "allocate_mc_bohb"
                   and (row.get("human_feedback") or {}).get("decision") == "approve")
    allocations = deepcopy(preview["allocations"])
    for child in allocations:
        if not Path(child["structure_path"]).is_file():
            raise FileNotFoundError(f"MC 结构文件不存在：{child['structure_path']}")
    mc_root = Path(plan["directory"])
    if plan["exists"]:
        if not allow_delete:
            raise ValueError("MC-search 仍存在，必须先确认删除")
        shutil.rmtree(mc_root)
    current = deepcopy(state)
    old_ids = set(plan["old_task_ids"])
    old_keys = {row.get("task_key") for row in current.get("tasks") or []
                if row.get("task_id") in old_ids}
    current["tasks"] = [row for row in current.get("tasks") or []
                        if row.get("task_id") not in old_ids]
    current["pending_tasks"] = [row for row in current.get("pending_tasks") or []
                                if row.get("task_id") not in old_ids]
    for key in old_keys:
        current.setdefault("budget_reservations", {}).pop(key, None)
    old_batch_ids = set(plan["old_batch_ids"])
    discarded_batches = [row for row in current.get("slurm_batches") or []
                         if row.get("batch_id") in old_batch_ids]
    current["slurm_batches"] = [row for row in current.get("slurm_batches") or []
                                if row.get("batch_id") not in old_batch_ids]
    for child in allocations:
        key = child["task_key"]
        if key in current["budget_reservations"]:
            raise ValueError(f"MC 预算预留与其他任务冲突：{key}")
        cost = float(child["planned_relative_cost"])
        current["budget_reservations"][key] = {
            "task_key": key, "stage": "deep_search", "reserved_cost": cost,
            "status": "reserved", "config_version": config_version,
            "model_version": child["model_version"], "reserved_at": None,
            "timeout_at": None, "settlement_id": None, "actual_cost": None,
            "released_cost": None, "approval_override": plan["allocation_checksum"]}
        task = {**child, "task_id": f"MC-{hashlib.sha256(key.encode()).hexdigest()[:12]}",
                "object_id": child["branch_id"], "status": "pending",
                "parameters": {"max_mc_steps": child["max_mc_steps"],
                               "patience_steps": child["patience_steps"],
                               "min_improvement": child["min_improvement"],
                               "seed": child["seed"]}}
        current["tasks"].append(task)
        current["pending_tasks"].append(task)
    tier_state = current.setdefault("tiered_mc_state", {})
    tier_state["segments"] = [row for row in tier_state.get("segments") or []
                              if row.get("task_key") not in old_keys] + allocations
    current.setdefault("mc_regeneration_history", []).append({
        **plan, "approved_action_task_key": next(row["final_action"]["task_key"]
            for row in reversed(current["action_records"])
            if row.get("status") == "completed"
            and (row.get("final_action") or {}).get("tool") == "allocate_mc_bohb"
            and (row.get("human_feedback") or {}).get("decision") == "approve"),
        "budget_override": "explicit_full_plan_approval",
        "discarded_batches": [{"batch_id": row.get("batch_id"),
                               "status": row.get("status"),
                               "upload_directory": row.get("upload_directory")}
                              for row in discarded_batches]})
    upload = prepare_mc_upload_batches(action={"parameters": {"mode": "mc_inputs"}},
        context={"effective_config": config, "event_state": current,
                 "manager": manager, "phase_references": phase_references})
    if upload["status"] != "prepared" or upload["task_count"] != len(allocations):
        raise RuntimeError(f"MC 输入未完整生成：{upload.get('status')} / {upload.get('task_count')}")
    return upload

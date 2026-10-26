"""Rebuild the current approved MC upload inputs without submitting jobs."""

from copy import deepcopy
from collections import Counter
import hashlib
import math
from pathlib import Path
import json
import uuid

from phase_agent.tools.local.prepare_mc_upload_batches import prepare_mc_upload_batches
from phase_agent.tools.mc_batch_policy import mc_batch_limit
from phase_agent.tools.remote.batch_runner import _compatibility


def is_mc_regeneration_request(message):
    text = str(message or "").lower().replace(" ", "")
    return any(word in text for word in ("重新生成", "重新跑", "重跑", "重建")) and any(
        word in text for word in ("mc", "蒙特卡洛")
    )


def is_mc_regeneration_confirmation(message):
    return str(message or "").strip().lower().replace(" ", "") in {
        "确认重新生成当前轮mc任务",
        "确认重建当前轮mc任务",
    }


def _mc_operation_directory(input_path):
    batch = Path(input_path).resolve().parent.parent
    return batch.parent.parent if batch.parent.name == "inputs" else batch.parent


def _approved_mc_preview(state):
    """Use a frozen preview or the exact legacy execution receipt, never a fresh allocation."""
    pending_keys = {
        t.get("task_key")
        for t in state.get("tasks", [])
        if t.get("stage") == "deep_search" and t.get("status") == "pending"
    }
    history = [
        h
        for h in state.get("mc_regeneration_history") or []
        if not pending_keys
        or pending_keys.intersection(
            r.get("task_key") for r in (h.get("replacement_preview") or {}).get("allocations", [])
        )
    ]
    if history and history[-1].get("replacement_preview"):
        preview = history[-1]["replacement_preview"]
        tasks = {row.get("task_key"): row for row in state.get("tasks") or []}
        fields = ("branch_id", "model_version", "structure_path", "max_mc_steps", "seed")
        if any(
            row["task_key"] not in tasks
            or any(tasks[row["task_key"]].get(field) != row.get(field) for field in fields)
            for row in preview["allocations"]
        ):
            raise ValueError("重新规划的 MC 清单与当前任务不一致")
        return deepcopy(preview), "user_requested_replanning"
    records = [
        row
        for row in state.get("action_records") or []
        if row.get("status") in {"completed", "not_configured"}
        and (row.get("final_action") or {}).get("tool") == "allocate_mc_bohb"
        and (row.get("human_feedback") or {}).get("decision") == "approve"
    ]
    if not records:
        raise ValueError("没有已批准的当前轮 MC 分配方案")
    matching = [
        r
        for r in records
        if pending_keys.intersection(
            a.get("task_key")
            for a in (
                ((r.get("final_action") or {}).get("parameters") or {}).get("budget_preview") or {}
            ).get("allocations", [])
        )
        or pending_keys.intersection(
            a.get("task_key")
            for a in ((r.get("execution_result") or {}).get("result") or {}).get("actions", [])
        )
    ]
    record = (matching or records)[-1]
    action = record["final_action"]
    parameters = action.get("parameters") or {}
    if "budget_preview" in parameters:
        preview = parameters.get("budget_preview") or {}
        if record.get("status") != "completed":
            actual = {t.get("task_key"): t for t in state.get("tasks", [])}
            accepted = {
                t.get("task_key")
                for t in ((record.get("execution_result") or {}).get("result") or {}).get(
                    "actions", []
                )
            }
            fields = ("branch_id", "model_version", "structure_path", "max_mc_steps", "seed")
            if not preview.get("allocations") or any(
                a.get("task_key") not in accepted
                or a.get("task_key") not in actual
                or any(actual[a["task_key"]].get(f) != a.get(f) for f in fields)
                or (state.get("budget_reservations", {}).get(a["task_key"]) or {}).get("status")
                != "reserved"
                for a in preview["allocations"]
            ):
                raise ValueError("部分失败的 MC 分配缺少已批准且一致的任务/预算记录")
        return deepcopy(preview), "approved_budget_preview"
    execution = record.get("execution_result") or {}
    result = execution.get("result") or {}
    children = result.get("actions") or []
    if (
        execution.get("status") != "completed"
        or result.get("status") != "completed"
        or not children
    ):
        raise ValueError("旧 MC 方案没有完整执行分配记录；不能推测或重新分配")
    fields = (
        "branch_id",
        "structure_id",
        "structure_path",
        "model_version",
        "max_mc_steps",
        "patience_steps",
        "min_improvement",
        "seed",
        "planned_relative_cost",
    )
    tasks = {row.get("task_key"): row for row in state.get("tasks") or []}
    keys = set()
    cost = 0.0
    steps = 0
    for child in children:
        key = child.get("task_key")
        if not key or key in keys or child.get("stage") != "deep_search":
            raise ValueError("旧 MC 执行分配记录含重复任务或非法阶段")
        keys.add(key)
        if child.get("branch_id") not in (action.get("target_ids") or []):
            raise ValueError("旧 MC 执行分支不在原批准范围内")
        count = child.get("max_mc_steps")
        relative_cost = child.get("planned_relative_cost")
        if (
            type(count) is not int
            or count <= 0
            or isinstance(relative_cost, bool)
            or not isinstance(relative_cost, (int, float))
            or not math.isfinite(relative_cost)
            or relative_cost < 0
        ):
            raise ValueError("旧 MC 执行记录缺少有效步数或成本")
        current = tasks.get(key)
        if current is not None and (
            current.get("stage") != "deep_search"
            or any(current.get(field) != child.get(field) for field in fields)
        ):
            raise ValueError("旧 MC 执行分配与当前任务记录不一致")
        steps += count
        cost += relative_cost
    step_limit = parameters.get("mc_budget")
    cost_limit = action.get("budget")
    if (
        type(step_limit) is not int
        or steps > step_limit
        or isinstance(cost_limit, bool)
        or not isinstance(cost_limit, (int, float))
        or not math.isfinite(cost_limit)
        or cost > cost_limit + 1e-9
    ):
        raise ValueError("旧 MC 执行分配超出原批准的步数或成本范围")
    from phase_agent.tools.remote.integrity import payload_checksum

    allocations = deepcopy(children)
    return {
        "allocations": allocations,
        "selected_branch_count": len(allocations),
        "estimated_relative_cost": cost,
        "allocation_checksum": payload_checksum(allocations),
    }, "approved_execution_record"


def plan_mc_regeneration(state, upload_root, *, replacement_preview=None):
    """Validate the approved plan and exact current-round MC directory."""
    preview, allocation_source = _approved_mc_preview(state)
    original_keys = {row.get("task_key") for row in preview.get("allocations") or []}
    if replacement_preview is not None:
        preview = deepcopy(replacement_preview)
        allocation_source = "user_requested_replanning"
    allocations = preview.get("allocations") or []
    if not allocations or len(allocations) != preview.get("selected_branch_count"):
        raise ValueError("已批准的 MC 分配清单不完整")
    if preview.get("estimated_relative_cost") is None:
        raise ValueError("已批准方案缺少 MC 成本估计")
    keys = [row.get("task_key") for row in allocations]
    if len(set(keys)) != len(keys) or not all(keys):
        raise ValueError("已批准的 MC 分配清单含重复或空任务键")
    current = [
        row
        for row in state.get("tasks") or []
        if row.get("stage") == "deep_search" and row.get("task_key") in original_keys
    ]
    if any(row.get("status") not in {"pending"} or row.get("job_id") for row in current):
        raise ValueError("当前轮 MC 已提交、完成或状态不明，不能重生成")
    old_ids = {row.get("task_id") for row in current}
    root = Path(upload_root).resolve()
    paths = {_mc_operation_directory(row["input_path"]) for row in current if row.get("input_path")}
    if len(paths) != 1:
        raise ValueError("无法唯一确定当前轮 MC-search 目录")
    mc_root = paths.pop()
    registered = {
        Path(row["operation_directory"]).resolve()
        for row in state.get("slurm_batches") or []
        if row.get("calculation_group") == "MC-search"
        and row.get("operation_directory")
        and set(row.get("task_ids") or []) & old_ids
    }
    if (
        mc_root.name != "MC-search"
        and not (mc_root.name.startswith("allocation-") and mc_root.parent.name == "MC-search")
    ) or root not in mc_root.parents:
        if not (mc_root in registered and root in mc_root.parents) and not (
            (mc_root.name.startswith("MC-round-") or "_MC-round-" in mc_root.name)
            and mc_root.parent.name.startswith("Search-group-")
            and root in mc_root.parents
        ):
            raise ValueError("MC-search 目录不在配置的上传根目录内")
    other_tasks = [
        row
        for row in state.get("tasks") or []
        if row.get("task_id") not in old_ids
        and row.get("input_path")
        and mc_root in Path(row["input_path"]).resolve().parents
    ]
    if other_tasks:
        raise ValueError("此目录还包含其他 MC 分配，请先拆分目录；不能删除历史输入或结果")
    batches = [
        row
        for row in state.get("slurm_batches") or []
        if (
            row.get("calculation_group") == "MC-search"
            and Path(row.get("upload_directory") or "").resolve().parent == mc_root
        )
        or set(row.get("task_ids") or []) & old_ids
    ]
    if any(
        row.get("job_id") or row.get("status") not in {"prepared", "superseded_inputs"}
        for row in batches
    ):
        raise ValueError("MC 批次可能已提交，不能重生成")
    non_mc_batches = [row for row in state.get("slurm_batches") or [] if row not in batches]
    kept_ids = {row.get("batch_id") for row in non_mc_batches}
    remote_start = len(non_mc_batches) + 1
    if any(
        f"remote-{number:06d}" in kept_ids
        for number in range(remote_start, remote_start + len(allocations))
    ):
        raise ValueError("保留的其他阶段批次编号与重建后的 MC 批次冲突")
    input_directories = sorted(
        {
            Path(row["input_path"]).resolve().parent.parent
            for row in current
            if row.get("input_path")
        }
    )
    input_archives = {
        directory.with_name(directory.name + ".upload.zip") for directory in input_directories
    }
    storage_archive = mc_root / "inputs" / "all.inputs.zip"
    if storage_archive.is_file():
        input_archives.add(storage_archive)
    for row in current:
        result = Path(row["result_path"]) if row.get("result_path") else None
        if result is not None and (
            result.is_file() or result.with_name("task.finished.json").is_file()
        ):
            raise ValueError("发现本轮 MC 结果文件，不能覆盖")
    for directory in input_directories:
        directory.relative_to(mc_root)
        returned = mc_root / "results" / (directory.name + ".results.zip")
        if returned.is_file():
            raise ValueError("发现本轮 MC 结果包，请先回收，不能重生成")
    allowed = {
        "batch.snapshot.json",
        "full_na_structure.vasp",
        "GPU.sh",
        "initial.vasp",
        "manifest.json",
        "run_mlip_batch.py",
        "_handoff_mlip_batch.py",
        "_gpu_accounting.py",
        "_gpu_transfer_bundle.py",
        "_handoff_export_batch_result.py",
        "transfer_bundle.py",
        "run_mlip_task.py",
        "SHA256SUMS",
        "task.json",
        "UPLOAD_AND_SUBMIT.md",
    }
    from phase_agent.tools.remote.integrity import file_checksum

    input_checksums = {}
    for directory in input_directories:
        if directory.is_symlink() or (
            hasattr(directory, "is_junction") and directory.is_junction()
        ):
            raise ValueError("MC 输入中含链接，不能重生成")
        for path in directory.rglob("*"):
            if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
                raise ValueError("MC 输入中含链接，不能重生成")
            if path.is_file():
                if path.name not in allowed:
                    raise ValueError(f"MC 输入中含结果、日志或未知文件，不能替换：{path}")
                input_checksums[str(path)] = file_checksum(path)
    for archive in input_archives:
        if archive.is_symlink():
            raise ValueError("MC 输入包是链接，不能替换")
        if archive.is_file():
            input_checksums[str(archive)] = file_checksum(archive)
    branch_phase = {
        row.get("branch_id"): row.get("P") for row in state.get("branch_candidates") or []
    }
    strata = {}
    for row in allocations:
        label = (
            f"{row.get('tier') or 'unknown'}/{branch_phase.get(row.get('branch_id')) or 'unknown'}"
        )
        strata[label] = strata.get(label, 0) + 1
    configured_batch_size = (
        ((state.get("confirmed_config") or {}).get("supercomputer") or {}).get("batch_sizes") or {}
    ).get("deep_search", 20)
    batch_size = mc_batch_limit(configured_batch_size)
    compatibility_counts = Counter(_compatibility(row) for row in allocations)
    estimated_batches = sum(
        math.ceil(count / batch_size) for count in compatibility_counts.values()
    )
    return {
        **({"replacement_preview": preview} if replacement_preview is not None else {}),
        "input_directories": [str(path) for path in input_directories],
        "input_archives": sorted(str(path) for path in input_archives if path.is_file()),
        "input_file_checksums": input_checksums,
        "allocation_source": allocation_source,
        "directory": str(mc_root),
        "exists": mc_root.exists(),
        "old_task_ids": sorted(old_ids),
        "old_batch_ids": sorted(row.get("batch_id") for row in batches if row.get("batch_id")),
        "old_batch_count": len(batches),
        "remote_batch_start": remote_start,
        "mc_sampling_start": 1,
        "allocation_count": len(allocations),
        "mc_batch_size": batch_size,
        "estimated_batch_count": estimated_batches,
        "allocation_by_tier_phase": dict(sorted(strata.items())),
        "total_mc_steps": sum(int(row["max_mc_steps"]) for row in allocations),
        "allocation_checksum": preview.get("allocation_checksum"),
        "estimated_relative_cost": preview.get("estimated_relative_cost"),
    }


def regenerate_mc_inputs(
    state,
    *,
    approved_plan,
    upload_root,
    config,
    manager,
    phase_references,
    config_version,
    allow_delete=False,
    replacement_preview=None,
):
    plan = plan_mc_regeneration(state, upload_root, replacement_preview=replacement_preview)
    if plan != approved_plan:
        raise ValueError("MC 状态或目录已变化，请重新检查并确认")
    preview = replacement_preview
    if preview is None:
        preview, _ = _approved_mc_preview(state)
    allocations = deepcopy(preview["allocations"])
    for child in allocations:
        if not Path(child["structure_path"]).is_file():
            raise FileNotFoundError(f"MC 结构文件不存在：{child['structure_path']}")
    if plan["exists"] and not allow_delete:
        raise ValueError("MC 输入仍存在，必须先确认备份并替换")
    current = deepcopy(state)
    old_ids = set(plan["old_task_ids"])
    old_keys = {
        row.get("task_key") for row in current.get("tasks") or [] if row.get("task_id") in old_ids
    }
    current["tasks"] = [
        row for row in current.get("tasks") or [] if row.get("task_id") not in old_ids
    ]
    current["pending_tasks"] = [
        row for row in current.get("pending_tasks") or [] if row.get("task_id") not in old_ids
    ]
    for key in old_keys:
        current.setdefault("budget_reservations", {}).pop(key, None)
    old_batch_ids = set(plan["old_batch_ids"])
    discarded_batches = [
        row for row in current.get("slurm_batches") or [] if row.get("batch_id") in old_batch_ids
    ]
    current["slurm_batches"] = [
        row
        for row in current.get("slurm_batches") or []
        if row.get("batch_id") not in old_batch_ids
    ]
    for child in allocations:
        key = child["task_key"]
        if key in current["budget_reservations"]:
            raise ValueError(f"MC 预算预留与其他任务冲突：{key}")
        cost = float(child["planned_relative_cost"])
        current["budget_reservations"][key] = {
            "task_key": key,
            "stage": "deep_search",
            "reserved_cost": cost,
            "status": "reserved",
            "config_version": config_version,
            "model_version": child["model_version"],
            "reserved_at": None,
            "timeout_at": None,
            "settlement_id": None,
            "actual_cost": None,
            "released_cost": None,
            "approval_override": plan["allocation_checksum"],
        }
        task = {
            **child,
            "task_id": f"MC-{hashlib.sha256(key.encode()).hexdigest()[:12]}",
            "object_id": child["branch_id"],
            "status": "pending",
            "parameters": {
                "max_mc_steps": child["max_mc_steps"],
                "patience_steps": child["patience_steps"],
                **(
                    {"min_improvement": child["min_improvement"]}
                    if child.get("min_improvement") is not None
                    else {}
                ),
                "seed": child["seed"],
            },
        }
        current["tasks"].append(task)
        current["pending_tasks"].append(task)
    new_ids = [
        row["task_id"]
        for row in current["tasks"]
        if row.get("task_key") in {child["task_key"] for child in allocations}
    ]

    def rebind_receipt(row):
        ids = set(row.get("task_ids") or [])
        if not ids & old_ids:
            return row
        if not ids <= old_ids:
            raise ValueError("不能只替换同一 MC 分配中的部分任务")
        return {**row, "task_ids": new_ids, "allocation_checksum": preview["allocation_checksum"]}

    current["mc_second_round_allocations"] = [
        rebind_receipt(row) for row in current.get("mc_second_round_allocations") or []
    ]
    if current.get("mc_second_round_allocation"):
        current["mc_second_round_allocation"] = rebind_receipt(
            current["mc_second_round_allocation"]
        )
    tier_state = current.setdefault("tiered_mc_state", {})
    tier_state["segments"] = [
        row for row in tier_state.get("segments") or [] if row.get("task_key") not in old_keys
    ] + allocations
    current.setdefault("mc_regeneration_history", []).append(
        {
            **plan,
            "approved_action_task_key": next(
                row["final_action"]["task_key"]
                for row in reversed(current["action_records"])
                if row.get("status") in {"completed", "not_configured"}
                and (row.get("final_action") or {}).get("tool") == "allocate_mc_bohb"
                and (row.get("human_feedback") or {}).get("decision") == "approve"
            ),
            "budget_override": "explicit_full_plan_approval",
            "discarded_batches": [
                {
                    "batch_id": row.get("batch_id"),
                    "status": row.get("status"),
                    "upload_directory": row.get("upload_directory"),
                }
                for row in discarded_batches
            ],
        }
    )
    return _prepare_with_input_backup(
        current, state, plan, upload_root, config, manager, phase_references
    )


def _prepare_with_input_backup(
    current, original, plan, upload_root, config, manager, phase_references
):
    """Archive exact input paths and restore them if preparation fails; never remove results."""
    operation = Path(plan["directory"]).resolve()
    root = Path(upload_root).resolve()
    sources = [Path(p) for p in [*plan["input_directories"], *plan["input_archives"]]]
    for source in sources:
        source.resolve().relative_to(operation)
        if source.resolve() == operation:
            raise ValueError("不能替换整个 MC 轮次目录")
    backup = operation.parent / ".mc_input_history" / operation.name / uuid.uuid4().hex
    backup.resolve().relative_to(root)
    from phase_agent.tools.remote.transfer_bundle import local_path

    backup = local_path(backup)
    parents = {source.parent for source in sources}
    previous_children = {path for parent in parents if parent.is_dir() for path in parent.iterdir()}
    backup.mkdir(parents=True)
    moved = []
    old_ids = set(plan["old_task_ids"])
    record = {
        "plan": plan,
        "tasks": [r for r in original.get("tasks") or [] if r.get("task_id") in old_ids],
        "batches": [
            r
            for r in original.get("slurm_batches") or []
            if r.get("batch_id") in plan["old_batch_ids"]
        ],
    }
    (backup / "record.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    try:
        for source in sources:
            if source.exists():
                target = backup / source.relative_to(operation)
                target.parent.mkdir(parents=True, exist_ok=True)
                source.rename(target)
                moved.append((source, target))
        current["mc_regeneration_history"][-1]["input_backup_directory"] = str(backup)
        upload = prepare_mc_upload_batches(
            action={"parameters": {"mode": "mc_inputs"}},
            context={
                "effective_config": config,
                "event_state": current,
                "manager": manager,
                "phase_references": phase_references,
            },
        )
        if upload["status"] != "prepared" or upload["task_count"] != plan["allocation_count"]:
            raise RuntimeError(
                f"MC 输入未完整生成：{upload.get('status')} / {upload.get('task_count')}"
            )
        if any(Path(path).name == "all.inputs.zip" for path in plan["input_archives"]):
            from phase_agent.tools.remote.submission_storage import pack_storage_directory

            pack_storage_directory(operation / "inputs", "inputs")
        return upload
    except Exception:
        new_children = {
            path for parent in parents if parent.is_dir() for path in parent.iterdir()
        } - previous_children
        replaced_sources = {source for source, _ in moved if source.exists()}
        for path in sorted(new_children | replaced_sources):
            path.resolve().relative_to(operation)
            failed = backup / "failed_new_inputs" / path.relative_to(operation)
            failed.parent.mkdir(parents=True, exist_ok=True)
            path.rename(failed)
        for source, archived in reversed(moved):
            source.parent.mkdir(parents=True, exist_ok=True)
            archived.rename(source)
        raise

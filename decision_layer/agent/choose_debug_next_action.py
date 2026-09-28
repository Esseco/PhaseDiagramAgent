"""Choose the safe next preparation step from confirmed local search state."""

import hashlib
import json
from pathlib import Path


def choose_debug_next_action(state, manager, config, *, allowed_tools, user_message="", target_branch_ids=None):
    text = str(user_message or "").lower()
    if any(word in text for word in ("生成branch", "生成 branch", "新增branch", "重新生成", "generate branch")):
        return None
    if "prepare_local_batch_files" not in allowed_tools or manager is None:
        return None
    mc_pending = [row for row in state.get("tasks") or [] if row.get("stage") == "deep_search"
                  and row.get("status") == "pending" and not row.get("slurm_batch_id")]
    if mc_pending:
        if any(not row.get("structure_id") or not row.get("structure_path") for row in mc_pending):
            return None
        digest = hashlib.sha256(json.dumps(sorted(row["task_id"] for row in mc_pending)).encode()).hexdigest()[:16]
        return {"tool": "prepare_local_batch_files", "task_key": f"prepare-mc-inputs:{digest}",
                "target_ids": [row["task_id"] for row in mc_pending],
                "parameters": {"mode": "mc_inputs"}, "budget": 0.0,
                "reason": "已分配 MC 预算且有下载后的 Relax 结构；先生成可上传 MC 输入，不在本地运行。",
                "expected_purpose": f"为 {len(mc_pending)} 个 MC task 准备最多20个任务一组的上传批次。",
                "decision_source": "debug_state_gate"}
    if (state.get("dedup_gate") or {}).get("status") != "ready":
        return None
    if not (config.get("upload_batches_directory") and (config.get("mlip") or {}).get("model_path")):
        return None
    from execution_layer.budget.estimate_stage_cost import estimate_stage_cost

    budget = config.get("budgets") or {}
    limit = budget.get("total_relative_cost")
    remaining = (float(limit) - float((state.get("budget_usage") or {}).get("total_relative_cost", 0) or 0)
                 - float(state.get("reserved_relative_cost", 0) or 0)) if limit is not None else float("inf")
    stage_limit = (budget.get("stage_limits") or {}).get("relax_and_feature") or {}
    stage_used = (((state.get("budget_usage") or {}).get("stages") or {}).get("relax_and_feature") or {})
    active = [row for row in (state.get("budget_reservations") or {}).values()
              if row.get("stage") == "relax_and_feature" and row.get("status") in {"reserved", "submitted", "running"}]
    if stage_limit.get("max_cost") is not None:
        remaining = min(remaining, float(stage_limit["max_cost"]) - float(stage_used.get("cost", 0) or 0)
                        - sum(float(row.get("reserved_cost", 0) or 0) for row in active))
    tasks_left = (int(stage_limit["max_tasks"]) - int(stage_used.get("tasks", 0) or 0) - len(active)
                  if stage_limit.get("max_tasks") is not None else float("inf"))
    count = min(3, int((config.get("bohb") or {}).get("relax_structures_per_branch", 3)))
    selected, cost, structures = [], 0.0, 0
    branches = manager.data.get("branches") or {}
    records = manager.data.get("structures") or {}
    version = (config.get("mlip") or {}).get("version") or (config.get("mlip") or {}).get("name")
    settings = (config.get("mlip") or {}).get("relax_parameters") or {}
    settings_id = hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()[:12]
    existing_keys = {row.get("task_key") for row in (state.get("tasks") or [])}
    requested = list(dict.fromkeys(target_branch_ids or []))
    if requested and any(branch_id not in branches for branch_id in requested):
        return None
    candidates = [(branch_id, branches[branch_id]) for branch_id in requested] if requested else list(branches.items())
    valid_ids = set((state.get("dedup_gate") or {}).get("valid_structure_ids") or [])
    for bid, branch in sorted(candidates, key=lambda item: (item[1].get("det_H") or 0, item[1].get("P") or "", item[0])):
        ids = [sid for sid in branch.get("structure_ids") or []
               if sid in records and (not valid_ids or sid in valid_ids)
               and (records[sid].get("metadata") or {}).get("initialization_method")
               == "electrostatic_top10_random3_layer_occupied"][:count]
        ids = [sid for sid in ids if f"relax-screen:{version}:{settings_id}:{sid}" not in existing_keys]
        if not ids or any(not Path(records[sid].get("source_path") or "").is_file() for sid in ids):
            continue
        branch_cost = sum(estimate_stage_cost(
            "relax_and_feature", atom_count=sum((records[sid].get("composition") or {}).values()),
            budgets=budget)["value"] for sid in ids)
        if cost + branch_cost > remaining or structures + len(ids) > tasks_left:
            continue
        selected.append(bid)
        cost += branch_cost
        structures += len(ids)
    if not selected:
        return None
    identity = json.dumps([config.get("config_version"), selected], sort_keys=True).encode()
    return {"tool": "prepare_local_batch_files", "target_ids": selected,
            "task_key": "prepare-relax-inputs:" + hashlib.sha256(identity).hexdigest()[:16],
            "parameters": {"mode": "relax_inputs", "selection_scope": "all_registered"}, "budget": round(cost, 6),
            "reason": f"已去重并保存 {structures} 个初态；调试模式先准备可上传的 MACE Relax 输入，不在本地计算或提交。",
            "expected_purpose": f"为 {len(selected)} 个 branch 准备 {structures} 个 Relax 任务及批量提交文件。",
            "decision_source": "debug_state_gate"}

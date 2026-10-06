"""Choose the safe next preparation step from confirmed local search state."""

import hashlib
import json
from fractions import Fraction
from pathlib import Path


def choose_debug_next_action(state, manager, config, *, allowed_tools, user_message="", target_branch_ids=None):
    text = str(user_message or "").lower()
    if any(word in text for word in ("生成branch", "生成 branch", "新增branch", "重新生成", "generate branch")):
        return None
    if "prepare_local_batch_files" not in allowed_tools or manager is None:
        return None
    dft_pending = [row for row in state.get("tasks") or []
                   if row.get("stage") in {"dft_relax", "dft_single_point"}
                   and row.get("status") == "pending" and not row.get("slurm_batch_id")
                   and row.get("recovery_wait_waived") is not True]
    if dft_pending:
        digest = hashlib.sha256(json.dumps(sorted(row["task_id"] for row in dft_pending)).encode()).hexdigest()[:16]
        return {"tool": "prepare_local_batch_files", "task_key": f"prepare-dft-inputs:{digest}",
                "target_ids": [row["task_id"] for row in dft_pending],
                "parameters": {"mode": "dft_inputs"}, "budget": 0.0,
                "reason": "为已批准的 DFT 任务准备 Py-Code atomate2 单点或优化输入。",
                "expected_purpose": "仅准备上传文件，不计算、不提交。",
                "decision_source": "debug_state_gate"}
    mc_pending = [row for row in state.get("tasks") or [] if row.get("stage") == "deep_search"
                  and row.get("status") == "pending" and not row.get("slurm_batch_id")]
    if mc_pending:
        if any(not row.get("structure_path") or not Path(row["structure_path"]).is_file()
               for row in mc_pending):
            return None
        digest = hashlib.sha256(json.dumps(sorted(row["task_id"] for row in mc_pending)).encode()).hexdigest()[:16]
        return {"tool": "prepare_local_batch_files", "task_key": f"prepare-mc-inputs:{digest}",
                "target_ids": [row["task_id"] for row in mc_pending],
                "parameters": {"mode": "mc_inputs"}, "budget": 0.0,
                "reason": "已分配 MC 预算且输入结构已回传；准备器核对 branch 引用并生成对应分配的上传文件。",
                "expected_purpose": f"为 {len(mc_pending)} 个 MC task 准备最多10个任务一组的上传批次。",
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
    mlip_config = config.get("mlip") or {}
    settings = mlip_config.get("relax_parameters") or {}
    settings_id = hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()[:12]
    existing_keys = {row.get("task_key") for row in (state.get("tasks") or [])}
    reusable_relax_ids = _verified_migrated_relax_ids(state, mlip_config, version)
    requested = list(dict.fromkeys(target_branch_ids or []))
    if requested and any(branch_id not in branches for branch_id in requested):
        return None
    candidates = [(branch_id, branches[branch_id]) for branch_id in requested] if requested else list(branches.items())
    # Once a versioned Relax hull has enough complete branches for MC, advance
    # that batch. Newly added/unrelaxed branches are supplementary work and
    # must not send an already-screened run back to Relax preparation.
    mc_action = _mc_action_from_completed_relax(
        state, candidates, records, config, version, count, allowed_tools,
    )
    if mc_action is not None:
        return mc_action
    has_mc = any(row.get("stage") == "deep_search" for row in state.get("tasks") or [])
    if has_mc:
        from scientific_layer.mc.second_round_state import (
            first_round_source, second_round_already_allocated,
        )
        source = first_round_source(state, version)
        second_enabled = (config.get("mc_policy") or {}).get("second_segment_enabled", True)
        if (not second_enabled or source is None
                or not second_round_already_allocated(state, version, source["checksum"])):
            # Supplementary, unscreened branches must not send an MC run back to Relax.
            return None
    valid_ids = set((state.get("dedup_gate") or {}).get("valid_structure_ids") or [])
    for bid, branch in sorted(candidates, key=lambda item: (item[1].get("det_H") or 0, item[1].get("P") or "", item[0])):
        ids = [sid for sid in branch.get("structure_ids") or []
               if sid in records and (not valid_ids or sid in valid_ids)
               and (records[sid].get("metadata") or {}).get("initialization_method")
               == "electrostatic_top10_random3_layer_occupied"][:count]
        ids = [sid for sid in ids
               if sid not in reusable_relax_ids
               and f"relax-screen:{version}:{settings_id}:{sid}" not in existing_keys]
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


def _verified_migrated_relax_ids(state, mlip_config, model_version):
    """Reuse only legacy Relax results explicitly verified by an approved migration."""
    current_config = state.get("confirmed_config_version")
    settings = mlip_config.get("relax_parameters") or {}
    expected = {
        "mace_head": mlip_config.get("mace_head") or settings.get("mace_head"),
        "fmax": settings.get("fmax"),
        "relax_cell": settings.get("relax_cell"),
        "relax_steps": settings.get("relax_steps"),
    }
    # Scheduling-only revisions do not invalidate scientific results. Walk
    # backwards through that audited chain before checking the original
    # legacy-result compatibility migration.
    trusted_versions = {current_config}
    migrations_list = list(state.get("config_migrations") or [])
    changed = True
    while changed:
        changed = False
        for migration in migrations_list:
            if (migration.get("type") == "confirmed_generation_policy_revision"
                    and migration.get("to") in trusted_versions
                    and migration.get("from") not in trusted_versions):
                trusted_versions.add(migration.get("from"))
                changed = True
    migrations = {}
    for migration in migrations_list:
        compatibility = migration.get("compatibility") or {}
        verified = compatibility.get("verified_result_fields") or {}
        if (migration.get("type") != "approved_migration_with_verified_legacy_mace_defaults"
                or migration.get("to") not in trusted_versions
                or compatibility.get("basis") != "verified_legacy_mace_mh_1_worker_defaults"
                or compatibility.get("model") != model_version
                or compatibility.get("legacy_config_version") != migration.get("from")
                or model_version != "mace-mh-1"
                or verified.get("mace_head") != expected["mace_head"]
                or verified.get("cell_relaxed") is not expected["relax_cell"]):
            continue
        try:
            if abs(float(verified.get("fmax_target_ev_per_angstrom")) - float(expected["fmax"])) > 1e-12:
                continue
            if int(verified.get("relax_steps_used_max")) > int(expected["relax_steps"]):
                continue
        except (TypeError, ValueError):
            continue
        migrations[migration.get("from")] = migration
        trusted_versions.add(migration.get("from"))

    reusable = set()
    for task in state.get("tasks") or []:
        if (task.get("config_version") not in trusted_versions
                or task.get("status") != "completed"
                or task.get("stage") not in {"relax_and_feature", "mlip_relax", "relax_screen"}
                or task.get("model_version") != model_version
                or not task.get("structure_id")):
            continue
        output = task.get("outputs") or task.get("result") or {}
        if (output.get("status") not in (None, "completed")
                or output.get("mace_head") != expected["mace_head"]
                or output.get("cell_relaxed") is not expected["relax_cell"]
                or output.get("relax_stopped_normally") is not True):
            continue
        try:
            if abs(float(output.get("fmax_target_ev_per_angstrom")) - float(expected["fmax"])) > 1e-12:
                continue
            if int(output.get("relax_steps_used")) > int(expected["relax_steps"]):
                continue
        except (TypeError, ValueError):
            continue
        structure_path = output.get("structure_path") or task.get("result_path")
        if structure_path and Path(structure_path).is_file():
            reusable.add(task["structure_id"])
    return reusable


def _mc_action_from_completed_relax(state, candidates, structures, config, version, count, allowed_tools):
    """Advance to MC only when the saved Relax pool covers each selected initial state."""
    if "allocate_mc_bohb" not in allowed_tools:
        return None
    from scientific_layer.mc.second_round_state import first_round_source, second_round_already_allocated
    has_mc = any(row.get("stage") == "deep_search" for row in state.get("tasks") or [])
    source = first_round_source(state, version) if has_mc else None
    if has_mc and (source is None or not (config.get("mc_policy") or {}).get("second_segment_enabled", True)
                   or second_round_already_allocated(state, version, source["checksum"])):
        return None
    hull_version = state.get("current_branch_hull_version")
    pool = (state.get("branch_hull_batches") or {}).get(hull_version) or {}
    if pool.get("model_version") != version:
        return None
    diagram = (state.get("phase_diagrams") or {}).get("mlip") or {}
    if (diagram.get("status") != "completed" or diagram.get("model_version") != version
            or not diagram.get("version")):
        return None
    ready_ids = {row.get("structure_id") for row in pool.get("records") or []
                 if row.get("model_version") == version
                 and row.get("structure_path") and Path(row["structure_path"]).is_file()}
    eligible = []
    for branch_id, branch in candidates:
        try:
            if Fraction(str(branch.get("x") or 0)) <= 0:
                continue
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        ids = [sid for sid in branch.get("structure_ids") or []
               if sid in structures and (structures[sid].get("metadata") or {}).get("initialization_method")
               == "electrostatic_top10_random3_layer_occupied"][:count]
        if ids and all(sid in ready_ids for sid in ids):
            eligible.append(branch_id)
    if not eligible:
        return None
    if source:
        first_ids = set(source["branch_ids"])
        eligible = [branch_id for branch_id in eligible if branch_id in first_ids]
        if not eligible:
            return None
    strategy = config.get("round_strategy") or {}
    raw_budget = (strategy.get("rule_default") or {}).get("mc_budget")
    if isinstance(raw_budget, bool) or not isinstance(raw_budget, (int, float)) or raw_budget <= 0:
        return None
    intent = state.get("mc_budget_intent") or {}
    requested_steps = intent.get("steps")
    mc_budget = (int(requested_steps) if isinstance(requested_steps, int)
                 and not isinstance(requested_steps, bool) and requested_steps > 0
                 else int(raw_budget))
    maximum = strategy.get("maximum_mc_budget")
    if maximum is not None:
        mc_budget = min(mc_budget, int(maximum))
    if mc_budget <= 0:
        return None
    from decision_layer.strategy.estimate_branch_mc_budget import estimate_branch_mc_budget
    seed = int(config.get("seed", (config.get("run") or {}).get("seed", 42)))
    preview = estimate_branch_mc_budget(
        [{**branch, "branch_id": bid} for bid, branch in candidates if bid in eligible],
        pool, state, config, step_limit=mc_budget, seed=seed)
    if not preview["allocations"]:
        return None
    missing_ids = set(preview.get("missing_branch_ids") or [])
    dispatch_branch_ids = [branch_id for branch_id in eligible if branch_id not in missing_ids]
    if not dispatch_branch_ids:
        return None
    missing_count = len(missing_ids)
    exploration = max(float((config.get("mc_policy") or {}).get("random_exploration_fraction", 0.1)),
                      float(strategy.get("minimum_exploration_fraction", 0.1)))
    identity = json.dumps([version, hull_version, eligible, mc_budget,
                           source["checksum"] if source else None], sort_keys=True).encode()
    return {"tool": "allocate_mc_bohb", "target_ids": dispatch_branch_ids,
            "task_key": "allocate-mc:" + hashlib.sha256(identity).hexdigest()[:16],
            "parameters": {"mc_budget": mc_budget, "dft_budget": 0,
                           "exploration_fraction": exploration, "seed": seed,
                           "hull_reference_version": hull_version,
                           "phase_diagram_version": diagram["version"],
                           "round_kind": "second" if source else "first",
                           "first_round_source_checksum": source["checksum"] if source else None,
                           "budget_preview": preview},
            "budget": preview["estimated_relative_cost"],
            "reason": ("首轮 MC 已完成；结合首轮结果、当前相图 Ehull、相分层和预算，提出一次第二轮 MC 分配。"
                       + (f"当前相图缺少唯一 Ehull/atom 匹配的 {missing_count} 个 branch，已排除。"
                          if missing_count else "")
                       if source else "已回收本版本 Relax 结果并建立结构能量池；按 Ehull 与结构间能量差分配首段 MC 预算。"),
            "expected_purpose": ((f"首轮已完成 {source['completed_count']} 个 branch；" if source else "已构建本版 MLIP 凸包；")
                + f"从 {len(eligible)} 个含 Na branch 中，"
                + (f"排除 {missing_count} 个缺少唯一相图 Ehull/atom 证据的 branch；" if missing_count else "")
                + f"完整方案需要 {preview['full_plan_steps']} 步，目标预算 {mc_budget} 步；"
                + ("超过目标，请选择完整运行并调整预算，或压缩方案。" if preview['exceeds_target'] else
                   f"可完整运行 {preview['selected_branch_count']} 个 branch，批准后准备 MC 文件。")),
            "decision_source": "debug_state_gate"}

"""One JSON per generation, with proposed bounds and actual branch inventory."""

import hashlib
import json
from pathlib import Path
from phase_agent.tools.relax_mc_batch.files import write, read
from phase_agent.tools.workflows.branch_review_view import readable_view, raw_record
from phase_agent.tools.policy.adaptive_branch_count import configured_selection_guards
from phase_agent.tools.workflows.compact_review import review_artifacts


def artifact_path(state, task_key, params, directory=None):
    config = state.get("confirmed_config") or {}
    root = (state.get("workspace_layout") or {}).get("workspace_root") or (
        config.get("storage") or (config.get("config") or {}).get("storage") or {}
    ).get("workspace_root")
    if root:
        identity = (
            str(task_key or json.dumps(params, sort_keys=True))
            + ":"
            + str(state.get("active_model_version"))
        )
        digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
        return Path(root) / "workflow_state/branch_plans" / f"branch_plan_{digest}.json"
    return Path(directory) / "branch_inventory.json" if directory else None


def proposal_artifacts(proposal, state):
    from phase_agent.tools.policy.region_sampling import is_region
    from phase_agent.tools.workflows.region_review import write_proposal

    if is_region(proposal.get("raw_action") or {}):
        return write_proposal(proposal, state)
    action = proposal.get("raw_action") or {}
    params = action.get("parameters") or proposal.get("action_parameters") or {}
    tool = action.get("tool") or proposal.get("recommended_action")
    if tool == "prepare_local_batch_files" and params.get("mode") == "relax_mc_inputs":
        return _relax_mc_scope_artifacts(state, action, params)
    if tool != "generate_branches":
        return {}
    path = artifact_path(state, action.get("task_key"), params)
    if path is None:
        return {}
    from phase_agent.tools.workflows.readonly_views import is_read_only

    if is_read_only():
        return review_artifacts(path, kind="branch", state=state)
    if path.exists():
        saved = read(path)
        original = raw_record(saved)
        if original.get("status") == "completed":
            if original is saved:
                write(path, readable_view(original), sort_keys=False)
            return review_artifacts(path, kind="branch", record=original, state=state)
    config = state.get("confirmed_config") or {}
    config = config.get("config") or config
    boundary = (
        config.get("boundary")
        or (config.get("system") or {}).get("boundary")
        or (config.get("generation") or {}).get("boundary")
        or {}
    )
    scope, cost = (
        params.get("module_execution_scope") or {},
        params.get("generation_cost_preview") or {},
    )
    payload = {
        "schema_version": 1,
        "status": "awaiting_approval",
        "task_key": action.get("task_key"),
        "model_version": state.get("active_model_version"),
        "config_version": state.get("confirmed_config_version"),
        "branch_pool_review": action.get("branch_pool_review"),
        "quantity_reason": (action.get("decision_review") or {}).get("rationale")
        or (action.get("scientific_decision") or {}).get("rationale")
        or action.get("reason"),
        "plan": {
            "phases": boundary.get("P"),
            "boundary": boundary,
            "max_det_H": params.get("max_det_H"),
            "supercells": [
                {"phase": phase, "H": matrix, "det_H": determinant(matrix)}
                for phase, matrices in (boundary.get("H") or {}).items()
                if isinstance(matrices, list)
                for matrix in matrices
                if determinant(matrix) is not None
                and (params.get("max_det_H") is None or determinant(matrix) <= params["max_det_H"])
            ],
            "Na_fractions": boundary.get("x") or "生成时按整数占位规则确定；端点相规则见 phases",
            "branch_count_policy": (
                action.get("decision_context") or proposal.get("decision_context") or {}
            ).get("branch_count_policy"),
            "selection_guards": configured_selection_guards(config),
            "candidate_quota": params.get("total_quota"),
            "selected_branch_upper": params.get("batch_size"),
            "initial_states_per_branch": params.get("initial_states_per_branch"),
            "seed": params.get("seed"),
            "allocations": [
                {
                    k: row[k]
                    for k in (
                        "phase",
                        "strategy",
                        "quota",
                        "min_det_H",
                        "max_det_H",
                        "na_min",
                        "na_max",
                        "reason",
                        "evidence_refs",
                        "x",
                    )
                    if k in row
                }
                for row in params.get("generation_plan") or []
            ],
            "quotas": params.get("quotas"),
            "note": "配额不等于去重后新增数量；各相/H矩阵/Na含量的实际分布在生成后登记。",
        },
        "execution_scope": "generate_and_prepare_relax_mc_inputs" if scope else "generate_only",
        "submit_authorized": False,
        "budget_preview": {
            k: cost[k] for k in ("status", "relax_mc_cost_upper", "remaining_budget") if k in cost
        },
        "actual": None,
    }
    write(path, readable_view(payload), sort_keys=False)
    return review_artifacts(path, kind="branch", record=payload, state=state)


def _relax_mc_scope_artifacts(state, action, params):
    preview = params.get("relax_mc_preview") or {}
    model_version = preview.get("model_version") or state.get("active_model_version")
    from phase_agent.tools.state.retired_generations import retired_task_keys

    retired = retired_task_keys(state)
    generation = next(
        (
            row
            for row in reversed(state.get("generation_history") or [])
            if row.get("model_version") == model_version
            and row.get("new_branch_ids")
            and row.get("task_key") not in retired
        ),
        None,
    )
    if generation is None:
        return {"scope_review_unavailable": True}
    paths = generation.get("artifacts") or {}
    generation_json = Path(paths.get("json") or "")
    if not generation_json.is_file():
        return {"scope_review_unavailable": True}

    record = raw_record(read(generation_json))
    summary = generation.get("summary") or {}
    sampling = summary.get("sampling_report") or {}
    generated = {str(value) for value in generation.get("new_branch_ids") or []}
    target_values = preview.get("branch_ids")
    if not isinstance(target_values, list):
        target_values = action.get("target_ids") or []
    selected = {str(value) for value in target_values if value}
    overlap = selected & generated
    mismatch = bool(selected and generated and not overlap)
    structures = preview.get("structures") or []
    workload = preview.get("workload") or {}
    structure_count = workload.get("relax_structure_tasks", len(structures))
    registered_structures = len(generation.get("registered_ids") or []) or summary.get(
        "registered_structures", 0
    )
    phase_counts = summary.get("phase_counts") or {}
    phase_text = ", ".join(f"{phase}: {count}" for phase, count in sorted(phase_counts.items()))
    det_range = summary.get("det_H_range") or []
    det_text = f"{det_range[0]}–{det_range[-1]}" if det_range else "未记录"

    lines = [
        "Branch 生成概况及 Relax+MC 范围核对",
        f"模型版本：{model_version}",
        f"Branch 生成批次：{generation.get('task_key') or '未记录'}",
        (
            f"生成结果：新增 {len(generated)} 个 branch，登记 {registered_structures} 个初始结构；"
            f"候选 {summary.get('proposed_branches', '未记录')}，"
            f"已登记相分布 {phase_text or '未记录'}，det(H) 范围 {det_text}。"
        ),
        (
            f"采样状态：{sampling.get('status', '未记录')}；停止原因："
            f"{sampling.get('stop_reason', '未记录')}；尝试 {sampling.get('attempts', '未记录')} 次。"
        ),
    ]
    regions = sampling.get("regions") or []
    if regions:
        lines.append("区域采样（区域 | 尝试 | 接受 | 重复）：")
        lines.extend(
            f"{row.get('region_id', '未记录')} | {row.get('attempts', '未记录')} | "
            f"{row.get('accepted', '未记录')} | {row.get('duplicates', '未记录')}"
            for row in regions
        )
    if summary.get("initialization_failed_branches") is not None:
        lines.append(f"初态初始化失败 branch：{summary['initialization_failed_branches']}。")
    generation_text = Path(paths.get("text") or "")
    if generation_text.is_file():
        lines.extend(["生成分布摘要：", *generation_text.read_text(encoding="utf-8").splitlines()])
    lines.extend(
        [
            "",
            f"本次 Relax+MC 目标：{len(selected)} 个 branch，{structure_count} 个 Relax 结构。",
            f"目标与本轮新增 branch 的交集：{len(overlap)}。",
            f"未出现在本轮新增清单中的目标 branch：{len(selected - generated)}。",
        ]
    )
    if selected - generated:
        lines.append("目标 branch ID：" + ", ".join(sorted(selected - generated)))
    if mismatch:
        lines.append(
            "范围结论：本批目标与本轮新增 branch 完全无交集；先核对并修正方案范围，再审批。"
        )
    elif selected and generated:
        lines.append(
            f"范围结论：本批纳入本轮新增 branch {len(overlap)}/{len(generated)} 个；"
            "未纳入的新增 branch 数量应结合方案理由核对。"
        )
    bundle_id = preview.get("bundle_id")
    batch = next(
        (
            row
            for row in (state.get("relax_mc_batches") or {}).values()
            if row.get("bundle_id") == bundle_id
        ),
        None,
    )
    if batch:
        verification = batch.get("verification") or {}
        lines.append(
            f"本地输入状态：{batch.get('status', '未记录')}；"
            f"远端环境核验：{verification.get('remote_environment', '未记录')}。"
        )
    lines.append(f"完整生成记录 JSON：{generation_json}")
    identity = ":".join(
        str(value or "")
        for value in (action.get("task_key"), generation.get("task_key"), preview.get("checksum"))
    )
    digest = hashlib.sha256(identity.encode()).hexdigest()[:12]
    report_path = generation_json.with_name(f"branch_relax_mc_scope_{digest}.txt")
    from phase_agent.tools.workflows.readonly_views import is_read_only

    if not is_read_only():
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    text_path = report_path if report_path.is_file() else generation_text
    return {
        "json": str(generation_json),
        "text": str(text_path),
        "scope_mismatch": mismatch,
        "scope_review_unavailable": not text_path.is_file(),
        "selected_branch_count": len(selected),
        "generated_branch_count": len(generated),
        "overlap_branch_count": len(overlap),
    }


def determinant(matrix):
    if (
        not isinstance(matrix, list)
        or len(matrix) != 3
        or any(not isinstance(r, list) or len(r) != 3 for r in matrix)
    ):
        return None
    a, b, c = matrix
    return abs(
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def result_artifacts(directory, history, manager, registered, state, action):
    from phase_agent.tools.policy.region_sampling import is_region

    if is_region(action):
        from phase_agent.tools.workflows.region_review import write_result

        return write_result(history, manager, registered, state, action)
    path = artifact_path(state, action.get("task_key"), action.get("parameters") or {}, directory)
    payload = (
        raw_record(read(path))
        if path.exists()
        else {"schema_version": 1, "task_key": action.get("task_key")}
    )
    new_ids, rows = set(history.get("new_branch_ids") or []), []
    for bid in sorted({r["branch_id"] for r in registered}):
        branch = manager.data["branches"][bid]
        structures = []
        for item in registered:
            if item["branch_id"] != bid:
                continue
            structure = (manager.data.get("structures") or {}).get(item["structure_id"]) or {}
            structures.append(
                {
                    "structure_id": item["structure_id"],
                    "path": structure.get("source_path"),
                    "atom_count": sum((structure.get("composition") or {}).values()) or None,
                }
            )
        rows.append(
            {
                "branch_id": bid,
                "is_new": bid in new_ids,
                "phase": branch.get("P"),
                "H": branch.get("H"),
                "supercell_det": determinant(branch.get("H")),
                "Na_fraction": str(branch["x"]) if branch.get("x") is not None else None,
                "composition": branch.get("composition"),
                "structure_count": len(structures),
                "structures": structures,
            }
        )
    payload.update(
        status="completed",
        actual={
            "new_branch_count": len(new_ids),
            "registered_branch_count": len(rows),
            "registered_structure_count": len(registered),
            "branches": rows,
        },
    )
    write(path, readable_view(payload), sort_keys=False)
    return review_artifacts(path, kind="branch", record=payload, state=state)

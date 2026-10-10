"""Read-only approval cards and factual revision differences."""

from phase_agent.tools.policy.file_approval import proposal_hash

LABELS = {
    "generate_branches": "生成候选结构",
    "prepare_local_batch_files": "准备计算输入",
    "select_dft_candidates": "筛选并准备 DFT 输入",
    "allocate_mc_bohb": "分配 MC 采样",
    "update_mlip": "准备模型微调",
    "adjust_strategy": "调整搜索策略",
    "check_convergence": "检查收敛",
    "pause_search": "暂停搜索",
}


def proposal_changes(record):
    proposal = record.get("agent_proposal") or {}
    history = record.get("feedback_history") or []
    previous = next(
        (
            item.get("prior_proposal")
            for item in reversed(history)
            if isinstance(item.get("prior_proposal"), dict)
        ),
        None,
    )
    if previous is None:
        return []
    changes = []
    old = previous.get("raw_action") or {}
    new = proposal.get("raw_action") or {}

    def compare(before, after, path):
        if before == after:
            return
        if isinstance(before, dict) and isinstance(after, dict):
            for key in sorted(set(before) | set(after)):
                compare(before.get(key), after.get(key), path + "." + key)
        else:
            changes.append({"field": path, "before": before, "after": after})

    for key in ("tool", "target_ids", "parameters", "budget", "reason", "expected_purpose"):
        compare(old.get(key), new.get(key), key)
    return changes


def review_card(plan_id, record):
    proposal = record.get("agent_proposal") or {}
    action = proposal.get("raw_action") or {}
    tool = action.get("tool") or proposal.get("recommended_action")
    params = action.get("parameters") or proposal.get("action_parameters") or {}
    preview = params.get("refresh_preview") or {}
    targets = action.get("target_ids") or []
    scope = f"已指定 {len(targets)} 个目标"
    if tool == "generate_branches" and params.get("total_quota") is not None:
        scope = f"计划采样 {params['total_quota']} 个候选结构"
    if preview:
        scope = f"首批刷新 {len(preview.get('candidates') or [])} 个结构；暂缓 {preview.get('deferred', '未记录')} 个"
    boundary = "完成本次动作后停止；后续新动作需另行审阅。"
    if tool in {"prepare_local_batch_files", "select_dft_candidates", "update_mlip"}:
        boundary = "完成本批输入准备后停止；提交超算属于后续操作。"
    if preview:
        boundary = (
            "首批及已展示上限内最多一轮补充输入生成；"
            f"补充成本上限 {preview.get('maximum_supplemental_cost', '未记录')}；超限另行审批。"
        )
    from phase_agent.runtime.chat_approval_rules import is_sensitive_proposal

    return {
        "plan_id": plan_id,
        "revision": record.get("revision", 0),
        "proposal_hash": proposal_hash(proposal),
        "title": LABELS.get(tool, tool or "待审阅动作"),
        "purpose": proposal.get("expected_purpose") or proposal.get("reason") or "未记录",
        "reason": proposal.get("reason") or "未记录",
        "scope": scope,
        "target_count": len(targets),
        "target_ids": targets,
        "estimated_cost": proposal.get("estimated_cost"),
        "budget": action.get("budget"),
        "approval_boundary": boundary,
        "sensitive": is_sensitive_proposal(proposal),
        "evidence_refs": proposal.get("evidence_refs") or action.get("evidence_refs") or [],
        "changes": proposal_changes(record),
    }


def revision_lines(record):
    changes = proposal_changes(record)
    if not changes:
        return []
    import json
    from phase_agent.runtime.turn_process import clean

    def short(value):
        text = json.dumps(clean(value), ensure_ascii=False)
        return text if len(text) <= 100 else text[:97] + "…"

    return [
        "本次修改（旧值 → 新值）：",
        *[
            f"- {row['field']}：{short(row['before'])} → {short(row['after'])}"
            for row in changes[:8]
        ],
        "旧版批准不适用于新版方案。",
    ]

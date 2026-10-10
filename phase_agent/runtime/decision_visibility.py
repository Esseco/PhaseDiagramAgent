"""Read-only projection of stated scientific decisions, never inferred reasoning."""

NAMES = {
    "generate_branches": "生成初始结构或扩展搜索",
    "select_dft_candidates": "补充 DFT",
    "update_mlip": "准备微调方案",
    "allocate_mc_bohb": "分配 MC 搜索",
    "prepare_local_batch_files": "准备计算输入",
    "check_convergence": "检查收敛",
    "pause_search": "暂停搜索",
    "adjust_strategy": "修订策略",
}


def decision_summary(action):
    if not isinstance(action, dict) or not (action.get("tool") or action.get("action_type")):
        return {}
    tool = action.get("tool") or action.get("action_type")
    budget = action.get("round_budget_review") or {}
    dft = action.get("post_dft_review") or {}
    return {
        "tool": tool,
        "recommendation": NAMES.get(tool, tool),
        "reason": action.get("reason") or budget.get("reason") or "方案未登记判断理由",
        "assessments": {
            name: dft[key]
            for name, key in (
                ("微调取舍", "finetune_assessment"),
                ("补DFT取舍", "dft_assessment"),
                ("搜索取舍", "search_assessment"),
                ("收敛判断", "convergence_assessment"),
                ("数据局限", "limitations"),
            )
            if dft.get(key)
        },
        "alternatives": budget.get("alternatives") or [],
        "uncertainty": budget.get("uncertainty") or "",
        "evidence_refs": action.get("evidence_refs") or [],
    }


def pending_decision(state):
    records = list((state.get("pending_execution_policies") or {}).values())
    if len(records) != 1:
        return {}
    proposal = records[0].get("agent_proposal") or {}
    action = proposal.get("raw_action") or {}
    return decision_summary(action)


def render_decision_card(state):
    from html import escape
    import json

    decision = pending_decision(state)
    if not decision:
        return "<p>暂无唯一待审批科学方案。</p>"
    reason = str(decision["reason"])
    brief = reason if len(reason) <= 180 else reason[:180] + "…"
    detail = {
        k: v
        for k, v in decision.items()
        if k in {"assessments", "alternatives", "uncertainty", "evidence_refs"} and v
    }
    if brief != reason:
        detail["完整理由"] = reason
    return (
        "<p><b>推荐：</b>"
        + escape(str(decision["recommendation"]))
        + "</p>"
        + "<p><b>原因：</b>"
        + escape(brief)
        + "</p><p><b>状态：</b>待批准，尚未执行。</p>"
        + (
            "<details><summary>备选取舍与依据</summary><pre>"
            + escape(json.dumps(detail, ensure_ascii=False, indent=2))
            + "</pre></details>"
            if detail
            else ""
        )
    )

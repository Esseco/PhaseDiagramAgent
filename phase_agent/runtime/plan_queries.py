"""Inspect persisted pending plans without rerunning scientific decisions."""

from phase_agent.tools.workflows.readonly_views import read_only_view


def is_plan_query(message):
    text = str(message).lower().replace(" ", "").strip("？?。.!！")
    return text in {
        "看看方案",
        "看下方案",
        "看一下方案",
        "查看方案",
        "查看当前方案",
        "展示方案",
        "展示当前方案",
        "当前方案是什么",
        "方案内容",
        "方案呢",
        "看看计划",
        "查看计划",
        "showplan",
        "showtheplan",
    }


@read_only_view
def pending_plan_reply(state):
    pending = state.get("pending_execution_policies") or {}
    regeneration = [
        ("微调输入重生成", state.get("pending_finetune_regeneration")),
        ("MC输入重生成", state.get("pending_mc_regeneration")),
    ]
    regeneration = [(name, plan) for name, plan in regeneration if plan]
    from phase_agent.runtime.review_requests import additional_reviews

    reviews = additional_reviews(state)
    total = len(pending) + len(regeneration) + len(reviews)
    if not total:
        return "当前没有待确认方案。可说“继续”获取下一步建议。"
    if total > 1:
        entries = [
            f"{key}：{(row.get('agent_proposal') or {}).get('recommended_action') or '操作方案'}"
            for key, row in pending.items()
        ]
        entries.extend(name for name, _ in regeneration)
        entries.extend(f"{row['plan_id']}：{row['review_card']['title']}" for row in reviews)
        return "有多个待确认方案，请指定要查看哪个：\n" + "\n".join(f"- {item}" for item in entries)
    if reviews:
        row = reviews[0]
        card = row["review_card"]
        import os

        approval = (
            "请在本机审批页核对并批准："
            + f"http://127.0.0.1:{os.environ.get('PHASE_CONTROL_PORT', '8765')}/phase/approval"
            if card.get("sensitive") and row.get("review_kind") != "model_activation"
            else "核对后可回复‘同意’或‘拒绝’。"
        )
        return (
            "已保存方案：\n"
            f"{card['title']}：{card['scope']}\n依据：{row['reason']}\n"
            f"审批边界：{card['approval_boundary']}\n{approval}\n本次仅查看，未批准或执行。"
        )
    if regeneration:
        name, plan = regeneration[0]
        return (
            f"方案：{name}，保留原轮编号。\n目录：`{plan.get('directory', '未记录')}`\n"
            "需先核对旧任务是否已提交，再按重生成流程单独确认；本次仅查看。"
        )
    from phase_agent.runtime.workflow_reply_presentation import format_workflow_reply

    plan_id, record = next(iter(pending.items()))
    proposal = record.get("agent_proposal") or {}
    from phase_agent.analysis.state.project_authority import project_identity

    identity = record.get("project_identity")
    if identity and identity != project_identity(state):
        return (
            "已保存方案：\n"
            "暂不可批准：当前模型或生效设置与方案依据不同；需更新具体方案。未执行任务。"
        )
    if not proposal:
        return "待确认记录缺少具体方案，暂不可批准；请检查方案记录。未执行任务。"
    raw = proposal.get("raw_action") or {}
    from phase_agent.tools.policy.branch_conditions import branch_entry_errors

    prerequisites = branch_entry_errors(raw, state)
    if (
        "new_model_branch_generation_required" in prerequisites
        or "post_refresh_branch_generation_separate_review_required" in prerequisites
    ):
        return (
            "已保存方案：\n暂不可批准：新模型刷新后须先生成新branch并展示实际数量、分布和结构清单。"
            "原方案保留，但不能跳过此步骤执行。\n"
            "下一步：提出新的branch生成方案供审批；生成结果展示后，再单独审批Relax/MC方案。"
        )
    task_key = raw.get("task_key")
    effective = (state.get("effective_decisions") or {}).get(task_key) or {}
    blocked = effective.get("status") in {
        "reserved",
        "pending",
        "running",
        "completed",
        "reconciliation_required",
    }
    details = format_workflow_reply(
        {"status": "awaiting_approval", "agent_proposal": proposal, "state": state},
        "state.json",
    )
    if proposal.get("recommended_action") == "update_mlip":
        details = details.replace("建议：update_mlip", "建议：生成超算微调训练输入")
    from phase_agent.tools.state.execution_receipts import recovery_report

    stored = (state.get("confirmed_config") or {}).get("storage") or {}
    root = stored.get("workspace_root")
    unsettled = []
    if root:
        from pathlib import Path

        actual_state = Path(root) / (stored.get("paths") or {}).get(
            "state", "workflow_state/state.json"
        )
        unsettled = recovery_report(actual_state, state).get("unsettled") or []
    if blocked or unsettled:
        details = "\n".join(
            line
            for line in details.splitlines()
            if not any(token in line for token in ("同意", "批准", "审批页", "详细记录："))
        )
        return (
            f"已保存方案：\n{details}\n"
            + (
                "暂不可批准：存在未核对的中断执行回执。\n"
                if unsettled
                else f"暂不可批准：任务键 {task_key} 已有 {effective['status']} 记录。\n"
            )
            + "需先核对原任务与执行回执；本次仅展示方案，未批准或执行。"
        )
    return f"已保存方案：\n{details}\n本次仅查看，未批准或执行。"

"""Inspect persisted pending plans without rerunning scientific decisions."""


def is_plan_query(message):
    text = str(message).lower().replace(" ", "").strip("？?。.!！")
    return text in {"看看方案", "看下方案", "看一下方案", "查看方案", "查看当前方案",
                    "展示方案", "展示当前方案", "当前方案是什么", "方案内容", "方案呢",
                    "看看计划", "查看计划", "showplan", "showtheplan"}


def pending_plan_reply(state):
    pending = state.get("pending_execution_policies") or {}
    regeneration = [("微调输入重生成", state.get("pending_finetune_regeneration")),
                    ("MC输入重生成", state.get("pending_mc_regeneration"))]
    regeneration = [(name, plan) for name, plan in regeneration if plan]
    total = len(pending) + len(regeneration)
    if not total:
        return "当前没有待确认方案。可说“继续”获取下一步建议。"
    if total > 1:
        entries = [f"{key}：{(row.get('agent_proposal') or {}).get('recommended_action') or '操作方案'}"
                   for key, row in pending.items()]
        entries.extend(name for name, _ in regeneration)
        return "有多个待确认方案，请指定要查看哪个：\n" + "\n".join(f"- {item}" for item in entries)
    if regeneration:
        name, plan = regeneration[0]
        return (f"方案：{name}，保留原轮编号。\n目录：`{plan.get('directory', '未记录')}`\n"
                "需先核对旧任务是否已提交，再按重生成流程单独确认；本次仅查看。")
    proposal = next(iter(pending.values())).get("agent_proposal") or {}
    raw = proposal.get("raw_action") or {}
    tool = proposal.get("recommended_action") or raw.get("tool")
    names = {"update_mlip": "生成超算微调训练输入", "generate_branches": "生成branch",
             "select_dft_candidates": "准备DFT输入", "allocate_mc_bohb": "分配MC搜索",
             "prepare_local_batch_files": "准备计算输入", "reevaluate_candidates": "新模型结构刷新",
             "check_convergence": "评估收敛", "pause_search": "暂停搜索"}
    lines = [f"方案：{names.get(tool, tool or '动作未记录')}。"]
    purpose = proposal.get("expected_purpose") or proposal.get("reason")
    if purpose:
        lines.append(f"目的：{purpose}")
    if tool == "update_mlip":
        lines.append("批准后仅准备超算训练文件，不本机训练或自动激活模型。")
    else:
        cost = (proposal.get("estimated_cost") or {}).get("estimated_total_cost")
        if cost is not None:
            lines.append(f"预计相对成本：{cost:.3g}。" if isinstance(cost, (int, float)) else f"预计相对成本：{cost}。")
    if tool == "update_mlip" and state.get("finetune_input_conflict"):
        lines.append("已有原轮输入与当前参数或数据不一致，重生成仍须单独确认。")
    lines.append("本次仅查看，未批准或执行。")
    return "\n".join(lines)

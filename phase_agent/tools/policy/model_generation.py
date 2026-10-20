"""Explicit new-MLIP generation/search prerequisites; no strategy selection."""

from phase_agent.tools.state.training_pending_validation import training_pending_validation


def new_model_generation_facts(state):
    refresh = state.get("model_refresh") or {}
    version = state.get("active_model_version")
    post_refresh = bool(
        version
        and refresh.get("status") == "completed"
        and refresh.get("new_model_version") == version
        and refresh.get("old_model_version")
        and refresh["old_model_version"] != version
        and not training_pending_validation(state)
    )
    checksum = (refresh.get("plan") or {}).get("checksum")
    from phase_agent.tools.state.retired_generations import retired_task_keys

    retired = retired_task_keys(state)
    rows = [
        row
        for row in state.get("generation_history") or []
        if row.get("task_key") not in retired
        and row.get("model_version") == version
        and row.get("new_branch_ids")
        and row.get("registered_ids")
        and (not checksum or row.get("refresh_plan_checksum") == checksum)
    ]
    ids = sorted({bid for row in rows for bid in row["new_branch_ids"]})
    searched = any(
        task.get("model_version") == version
        and task.get("branch_id") in ids
        and task.get("stage") == "deep_search"
        and task.get("status") == "completed"
        for task in state.get("tasks") or []
    )
    return {
        "required": post_refresh,
        "post_refresh": post_refresh,
        "prefer_new_branches": True,
        "ready": bool(ids) if post_refresh else True,
        "model_version": version,
        "refresh_plan_checksum": checksum,
        "new_branch_count": len(ids),
        "generated_branch_ids": ids,
        "first_search_completed": searched,
        "instruction": (
            "新模型完成已有结构与相图刷新后，必须先生成当前模型的新branch。"
            "先展示生成方案并审批；生成后展示实际数量、分布与结构清单，再单独审批Relax/MC输入方案。"
            "生成方案不得捆绑后续输入准备；暂缓或空批次不满足本步骤，不设固定最低数量。"
            "完成本步骤后，再由Agent按需选择旧branch续搜、新增branch或DFT。"
            "新旧branch均可成为搜索候选；旧模型搜索历史不等于当前模型搜索已完成，"
            "基态/近简并与续搜规则保持有效，同模型已登记任务不得通过重新封装重置。"
            "撤回批次不作为有效新增或候选。刷新与审批、模型绑定、有效数据仍是硬约束。"
        ),
    }


def new_model_search_errors(state, branch_ids=None, *, dft=False):
    from phase_agent.tools.state.retired_generations import retired_branch_ids

    if branch_ids and set(branch_ids).intersection(retired_branch_ids(state)):
        return ["retired_generation_branch_not_available"]
    facts = new_model_generation_facts(state)
    if facts["required"] and not facts["ready"]:
        return ["new_model_branch_generation_required"]
    return []

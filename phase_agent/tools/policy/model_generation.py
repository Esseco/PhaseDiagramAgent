"""Explicit new-MLIP generation/search prerequisites; no strategy selection."""

from phase_agent.tools.state.training_pending_validation import training_pending_validation


def new_model_generation_facts(state):
    refresh = state.get("model_refresh") or {}
    version = state.get("active_model_version")
    required = bool(
        version
        and refresh.get("status") == "completed"
        and refresh.get("new_model_version") == version
        and refresh.get("old_model_version")
        and refresh["old_model_version"] != version
        and not training_pending_validation(state)
    )
    checksum = (refresh.get("plan") or {}).get("checksum")
    rows = [
        row
        for row in state.get("generation_history") or []
        if row.get("model_version") == version
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
        "required": required,
        "ready": bool(ids) if required else True,
        "model_version": version,
        "refresh_plan_checksum": checksum,
        "new_branch_count": len(ids),
        "generated_branch_ids": ids,
        "first_search_completed": searched,
        "instruction": (
            "模型轮次顺序：新MLIP激活→刷新已有结构/相图→生成并登记本轮新branch→Relax+MC搜索→按结果选择DFT。"
            "旧branch池可作为生成依据，不能代替本轮新增branch。尚未生成时先提出generate_branches具体方案；"
            "策略、数量与范围由Agent结合当前模型证据和预算选择，不自动执行，不机械用满配额。"
            "首次搜索只选generated_branch_ids；生成0个新branch不能宣称完成此阶段。"
            "明确绑定当前模型的独立验证请求另行保留，不能把历史验证计划当作默认顺序。"
        )
        if required
        else "当前不处于新模型刷新后的首次搜索阶段；此轮次前置约束不适用，沿用现有流程。",
    }


def new_model_search_errors(state, branch_ids=None, *, dft=False):
    facts = new_model_generation_facts(state)
    if not facts["required"]:
        return []
    plan = state.get("training_validation_planning") or {}
    scope = (plan.get("scope") or {}).get("model_version") or plan.get("model_version")
    if dft and plan.get("status") == "active" and scope == facts["model_version"]:
        return []  # Explicit current-model validation, not the default search route.
    if not facts["ready"]:
        return ["new_model_branch_generation_required"]
    if dft and not facts["first_search_completed"]:
        return ["new_model_branch_search_required_before_dft"]
    if branch_ids and not facts["first_search_completed"]:
        if not set(branch_ids).issubset(facts["generated_branch_ids"]):
            return ["first_new_model_search_requires_generated_branches"]
    return []

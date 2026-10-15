"""Model-scoped training evidence; no scientific action selection or permission."""

from collections import Counter
from copy import deepcopy


INACTIVE = {"historical", "completed", "cancelled", "superseded"}


def post_refresh_search_context(state):
    from phase_agent.tools.state.training_pending_validation import training_pending_validation

    refresh = state.get("model_refresh") or {}
    version = state.get("active_model_version")
    if training_pending_validation(state) or (
        not version
        or refresh.get("status") != "completed"
        or refresh.get("new_model_version") != version
    ):
        return {}
    refresh_ids = set(refresh.get("task_ids") or [])
    tasks = [
        t
        for t in state.get("tasks") or []
        if t.get("model_version") == version
        and t.get("task_id") not in refresh_ids
        and not t.get("model_refresh_id")
    ]
    counts = Counter((t.get("stage"), t.get("status")) for t in tasks)
    return {
        "status": "refresh_completed",
        "model_version": version,
        "old_model_version": refresh.get("old_model_version"),
        "available_branch_count": len(state.get("branch_candidates") or []),
        "mc_completed": counts[("deep_search", "completed")],
        "mc_pending": sum(
            counts[("deep_search", s)] for s in ("pending", "running", "submitted", "unknown")
        ),
        "relax_completed": counts[("relax_and_feature", "completed")],
        "instruction": (
            "当前模型已激活且结构与相图刷新完成。旧微调审阅和验证计划不是新轮次的执行指令。"
            "先生成本轮新branch，再进行Relax+MC搜索；是否完成由new_model_generation事实判断。"
            "旧branch池用于生成依据，不能把复用旧branch当作本轮新增。"
            "刷新任务不等于新一轮MC搜索；缺少独立验证不构成必须先补DFT的条件。"
            "若选择DFT，说明当前候选的具体价值以及为何此时优于branch搜索；不机械扩搜或花满预算。"
            "训练/CV误差仍不等于独立测试，未验证的精度保持未知。每次一个动作，照常审批。"
        ),
    }


def validation_planning_context(state):
    plan = state.get("training_validation_planning") or {}
    if not plan:
        return {}
    active = state.get("active_model_version")
    scope = (plan.get("scope") or {}).get("model_version") or plan.get("model_version")
    refresh = post_refresh_search_context(state)
    historical = (
        plan.get("status") in INACTIVE
        or bool(scope and active and scope != active)
        or bool(refresh and not scope)
    )
    if not historical:
        return {**deepcopy(plan), "status": "active", "model_version": scope or active}
    # Keep measured evidence and uncertainty; do not resend obsolete imperatives
    # such as no_activate, next_proposal, or an executable followup_action.
    evidence = [
        {
            k: deepcopy(row[k])
            for k in (
                "evaluation_type",
                "old",
                "new",
                "structures",
                "returned_structures",
                "training_fingerprint",
                "limitations",
            )
            if k in row
        }
        for row in plan.get("reviews") or []
    ]
    for row in evidence:
        if "old" in row:
            row["old_metrics"] = row.pop("old")
        if "new" in row:
            row["new_metrics"] = row.pop("new")
    return {
        "status": "historical",
        "model_version": scope,
        "active_model_version": active,
        "reviews": evidence,
        "instruction": "历史训练与验证证据，不能作为当前待办或强制DFT前置条件；独立验证是否完成仍按实际证据判断。",
    }


def training_review_active(context):
    if context.get("training_already_completed") and not context.get("post_refresh_search"):
        return True
    plan = context.get("training_validation_planning") or {}
    if plan:
        return plan.get("status") not in INACTIVE
    return bool(context.get("training_already_completed")) and not context.get(
        "post_refresh_search"
    )

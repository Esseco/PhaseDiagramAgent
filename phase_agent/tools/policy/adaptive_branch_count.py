"""Explicit current user policy, attached only to newly proposed generation actions."""

VERSION = "adaptive-branch-100-1000-v1"


def policy_for(state):
    from phase_agent.tools.policy.region_sampling import (
        VERSION as REGION_VERSION,
        policy,
    )

    if (state.get("branch_count_policy") or {}).get("version") == REGION_VERSION:
        return policy()
    context = state.get("decision_context") or {}
    branches = (
        state.get("available_branches")
        or state.get("branch_candidates")
        or context.get("available_branches")
        or []
    )
    return {
        "version": VERSION,
        "minimum": 100,
        "maximum": 1000,
        "first_round": not bool(branches),
        "budget_enforcement": "advisory",
        "instruction": "科学采样位置和数量由Agent根据当前项目证据、有效记忆、已确认人工约束及历史收益决定，代码不预设300个或固定策略比例。每轮选择100–1000个branch，说明采哪里、采多少及依据；程序只汇总数量和校验一致性，不代替科学取舍。每项给出相、合法超胞范围、Na范围、数量和简短理由；区分观察事实与探索假设，记忆为空就明确为空，不编造覆盖缺口或收益。首轮倾向更充分覆盖，后续自适应；优先小超胞，再按收益和成本扩大，不要求穷举小超胞；不机械填满1000或沿用旧默认。遵守已确认的结构边界、固定变量、当前模型、去重规则及审批；预算仅粗估，不单独截断数量。batch_size匹配整轮目标，不沿用96。有限合法空间、去重或已确认分组规则导致不足时如实报告，不擅自补算。方案仍需人工审批，提出方案不授予执行权限。",
    }


def is_adaptive_generation(action):
    return (action.get("tool") or action.get("action_type")) == "generate_branches" and (
        (action.get("decision_context") or {}).get("branch_count_policy") or {}
    ).get("version") == VERSION


def budget_limits_for_action(action, limits):
    return (
        {**(limits or {}), "enforcement": "advisory"} if is_adaptive_generation(action) else limits
    )


def configured_selection_guards(config):
    """Explicit configuration, not the generator's legacy small defaults."""
    run = config.get("run") or config
    options = run.get("generation_options") or config.get("generation_options") or {}
    selection = options.get("selection_config") or {}
    return {key: selection.get(key) for key in ("max_per_framework", "max_per_parent_branch")}


def generation_review_errors(action, state=None):
    """Reject stale or contradictory current-policy reviews; never revise them."""
    if (action.get("tool") or action.get("action_type")) != "generate_branches":
        return []
    from phase_agent.tools.policy.region_sampling import (
        VERSION as REGION_VERSION,
        is_region,
        errors as region_errors,
    )

    required_region = ((state or {}).get("branch_count_policy") or {}).get(
        "version"
    ) == REGION_VERSION
    if is_region(action):
        return region_errors(action.get("parameters") or {})
    if required_region:
        return ["region_sampling_requires_proposal_revision"]
    required = ((state or {}).get("branch_count_policy") or {}).get("version") == VERSION
    if required and not is_adaptive_generation(action):
        return ["branch_count_policy_requires_proposal_revision"]
    if not is_adaptive_generation(action):
        return []
    params = action.get("parameters") or {}
    total = params.get("total_quota")
    errors = []
    if type(total) is not int or not 100 <= total <= 1000:
        errors.append("branch_target_requires_100_to_1000")
    if params.get("batch_size") != total:
        errors.append("branch_target_mismatch")
    plan = params.get("generation_plan") or []
    if not plan:
        errors.append("branch_requires_explicit_allocation_plan")
    for index, row in enumerate(plan):
        if any(row.get(key) is None for key in ("na_min", "na_max", "min_det_H", "max_det_H")):
            errors.append(f"branch_allocation_bounds_missing:{index}")
    return errors

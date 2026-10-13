"""Current failures and feasible MC rows, never action selection or permission."""

from collections import Counter


def refresh_result_facts(state):
    refresh = state.get("model_refresh") or {}
    ids = set(refresh.get("task_ids") or [])
    rows = [t for t in state.get("tasks") or [] if t.get("task_id") in ids]
    errors = Counter(
        str(t.get("error") or t.get("failure_reason") or "原因未记录")[:600]
        for t in rows
        if t.get("status") in {"failed", "timeout"}
    )
    return {
        "status": refresh.get("status"),
        "task_count": len(ids),
        "registered_count": len(rows),
        "status_counts": dict(Counter(t.get("status", "unknown") for t in rows)),
        "failure_groups": [{"count": count, "reason": reason} for reason, count in errors.items()],
        "instruction": "失败是已回收的失败结果，不是仍在运行。先解释原因和修复依赖；本地生成输入只需远端模型路径，不需要本地权重；若错误来自远端加载，核对超算路径或在超算恢复文件。继续/同意不会恢复远端模型文件，也不能把失败标记为完成或自动重试。修复后重试仍需具体方案审批。",
    }


def mc_allocation_facts(state):
    """Expose the same scheduler constraints used by the approval preview."""
    from phase_agent.tools.policy.branch_conditions import mc_entry_facts
    from phase_agent.decisions.strategy.estimate_branch_mc_budget import (
        estimate_branch_mc_budget,
    )

    facts = mc_entry_facts(state)
    if not facts["ready"] or not state.get("branch_candidates"):
        return facts
    config = state.get("confirmed_config") or {}
    strategy = config.get("round_strategy") or {}
    steps = (strategy.get("rule_default") or {}).get("mc_budget", 900)
    if strategy.get("maximum_mc_budget") is not None:
        steps = min(int(steps), int(strategy["maximum_mc_budget"]))
    try:
        preview = estimate_branch_mc_budget(
            state["branch_candidates"],
            state["branch_hull_batches"][facts["hull_reference_version"]],
            state,
            config,
            step_limit=int(steps),
            seed=int((config.get("run") or {}).get("seed", 42)),
        )
    except (ValueError, KeyError, TypeError) as error:
        return {**facts, "allocation_error": str(error)}
    return {
        **facts,
        "round_kind": preview["round_kind"],
        "allocation_status": preview["status"],
        "available_allocations": [
            {
                k: row.get(k)
                for k in (
                    "branch_id",
                    "segment_index",
                    "max_mc_steps",
                    "planned_relative_cost",
                    "segment_reason",
                )
            }
            for row in preview["allocations"]
        ],
        "full_plan_steps": preview["full_plan_steps"],
        "full_plan_branch_count": preview["full_plan_branch_count"],
        "missing_branch_ids": preview["missing_branch_ids"],
        "allocation_instruction": "这是现有 MC 规则核算出的本轮可分配分支，不是要求执行。第二轮只延长满足晋级条件的已完成第一轮分支，不把未做第一轮的分支混入。LLM自行比较可行路线；allocate_mc_bohb 选择这里的实际 branch_id。无可分配项时解释条件并选择其他可行路线，不重复同一失败方案；预算是上限。",
    }


def convergence_feedback(state):
    for row in reversed(state.get("action_records") or []):
        action = row.get("final_action") or row.get("action") or {}
        if action.get("tool") != "check_convergence":
            continue
        execution = row.get("execution_result") or row.get("execution") or {}
        report = execution.get("result") or {}
        return {
            "record_id": row.get("record_id"),
            "status": report.get("status") or row.get("status"),
            "reason": report.get("reason"),
            "missing_evidence": report.get("missing_evidence") or [],
            "checks": report.get("checks") or {},
            "instruction": "这是上次已执行检查的结果。缺少不同模型轮次或DFT误差验证时，重复检查相同事实不能补出证据；结合新的有效结果选择可行下一步，不将缺证据误称程序运行中或已收敛。",
        }
    return {}

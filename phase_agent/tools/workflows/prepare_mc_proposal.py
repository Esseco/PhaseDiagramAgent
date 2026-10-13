"""One bounded model correction of invalid MC choices, always unapproved."""

from copy import deepcopy
from phase_agent.tools.workflows.prepare_mc_preview import prepare_mc_preview
from phase_agent.analysis.state.workflow_blockers import mc_allocation_facts
from phase_agent.tools.policy.execution_policy import build_agent_proposal
from phase_agent.tools.budget.record_budget_usage import record_budget_usage


def prepare_mc_proposal(action, state, decision_state, context, config, *, agent_client, revise):
    try:
        return prepare_mc_preview(action, state, context, config), state, None
    except ValueError as error:
        reason = str(error)
    facts = mc_allocation_facts(state)
    if not callable(agent_client):
        return None, state, reason + "；未配置模型修订，未执行任务。"
    if not facts.get("available_allocations"):
        return (
            None,
            state,
            reason + "。当前可分配范围为空；需评估其他路线，不重复相同MC方案。",
        )
    revision_state = deepcopy(decision_state)
    revision_state["decision_context"] = {
        "current_project": deepcopy(decision_state.get("current_project")),
        "mc_entry_conditions": facts,
        "long_term_memory": deepcopy(
            (decision_state.get("decision_context") or {}).get("long_term_memory") or {}
        ),
        "mc_validation_feedback": {
            "reason": reason,
            "instruction": "只修订本轮MC候选和步数，使用当前可分配分支；不执行、不继承旧批准。",
        },
    }
    revision_state.pop("available_branches", None)
    revision_state.pop("search_coverage", None)
    revision = revise(
        build_agent_proposal(action, decision_state),
        "本地MC核验未通过：" + reason + "。根据mc_entry_conditions返回可审批的新MC方案，不执行。",
        state=revision_state,
        allowed_tools=["allocate_mc_bohb"],
        agent_client=agent_client,
        source_state=state,
        manager=context.get("manager"),
        config=config,
    )
    usage = revision.get("llm_usage") or (revision.get("action") or {}).get("_llm_usage")
    if usage:
        state = record_budget_usage(
            state, {"llm_usage": usage, "iteration": state.get("iteration", 0)}
        )
    if revision.get("revision_status") != "revised":
        return None, state, reason + "；一次模型修订仍未成功，未执行任务。"
    try:
        revised = prepare_mc_preview(revision["action"], state, context, config)
    except ValueError as error:
        return None, state, str(error) + "；已停止本轮修订，未执行任务。"
    revised["parameters"]["plan_revision_notice"] = "原MC方案未通过核验；模型已修订，请审核新范围。"
    return revised, state, None

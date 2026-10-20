"""One model correction of an unapproved duplicate pipeline; no route selection."""

from phase_agent.tools.relax_mc_batch.plan import registration_conflicts
from phase_agent.runtime.turn_process import process_event


def propose_with_relax_feedback(state, *, agent_client, allowed_tools, config, propose):
    action = propose(state, agent_client=agent_client, allowed_tools=allowed_tools, config=config)
    facts = (state.get("decision_context") or {}).get("relax_mc_entry_conditions") or {}
    conflicts = registration_conflicts(action, facts)
    if not conflicts or not callable(agent_client):
        return action, None, None
    prior_usage = action.get("_llm_usage")
    feedback = {
        "code": "relax_inputs_already_registered",
        "conflicts": conflicts,
        "instruction": "原提案未执行、未获批准。根据这些真实任务冲突修订一项可行新提案；仍由你比较未冲突旧branch、新增branch及其他allowed_tools路线，不机械改选、不假设任务已删除或需要重算。保留用户目标与配置，不能继承批准；同一冲突不再重试。",
    }
    process_event("方案可执行性反馈", feedback)
    revision = {
        **state,
        "decision_context": {
            **(state.get("decision_context") or {}),
            "proposal_validation_feedback": feedback,
        },
    }
    revised = propose(
        revision, agent_client=agent_client, allowed_tools=allowed_tools, config=config
    )
    repeated = registration_conflicts(revised, facts)
    if repeated:
        return (
            revised,
            prior_usage,
            (
                f"Agent一次修订后仍选择已登记Relax任务的{len(repeated)}个branch；"
                "本轮已停止，未新建任务或继承批准。冲突见过程页的方案可执行性反馈。"
            ),
        )
    return revised, prior_usage, None

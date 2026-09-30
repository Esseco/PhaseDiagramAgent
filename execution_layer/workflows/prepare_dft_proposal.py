"""Preview and one bounded LLM revision, without execution or approval."""
from copy import deepcopy
from execution_layer.workflows.attach_dft_preview import attach_dft_preview
from execution_layer.policy.execution_policy import build_agent_proposal
from execution_layer.budget.record_budget_usage import record_budget_usage


def prepare_dft_proposal(action, state, decision_state, config, *, agent_client=None, revise):
    prepared, error = attach_dft_preview(action, state, config)
    if not error or agent_client is None:
        return prepared, state, error
    revision_state = deepcopy(decision_state)
    revision_state.setdefault("decision_context", {})["dft_validation_feedback"] = {
        "error": error, "instruction": "保留记忆与当前证据，修订原方案；不要由代码替选。仍需人工重新批准。"}
    revision = revise(build_agent_proposal(action, decision_state),
        "本地核验未通过：" + error + "。请自行重新权衡并返回合规DFT方案，不执行。",
        state=revision_state, allowed_tools=["select_dft_candidates"], agent_client=agent_client,
        source_state=state, config=config)
    usage = revision.get("llm_usage") or (revision.get("action") or {}).get("_llm_usage")
    if usage:
        state = record_budget_usage(state, {"llm_usage": usage, "iteration": state.get("iteration", 0)})
    if revision.get("revision_status") != "revised":
        return None, state, error + "；LLM修订失败，未执行任务。"
    prepared, error = attach_dft_preview(revision["action"], state, config)
    if prepared:
        prepared["parameters"]["plan_revision_notice"] = "原方案未通过核验；LLM已修订，需确认新方案。"
    return prepared, state, error

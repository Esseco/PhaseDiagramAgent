"""Preview and one bounded LLM revision, without execution or approval."""

from copy import deepcopy
from dataclasses import dataclass
from phase_agent.tools.workflows.attach_dft_preview import attach_dft_preview
from phase_agent.tools.policy.execution_policy import build_agent_proposal
from phase_agent.tools.budget.record_budget_usage import record_budget_usage


@dataclass(frozen=True)
class DFTProposalPreparation:
    action: dict
    state: dict
    error: str | None = None

    def __post_init__(self):
        if not isinstance(self.action, dict) or not self.action.get("tool"):
            raise ValueError("DFT preparation must retain a concrete proposal")
        if self.error is not None and (not isinstance(self.error, str) or not self.error):
            raise ValueError("Rejected DFT preparation requires a reason")

    @property
    def status(self):
        return "rejected" if self.error is not None else "ready"


def prepare_dft_proposal(action, state, decision_state, config, *, agent_client=None, revise):
    prepared, error = attach_dft_preview(action, state, config)
    if not error or agent_client is None:
        return DFTProposalPreparation(prepared if not error else deepcopy(action), state, error)
    revision_state = deepcopy(decision_state)
    revision_state.setdefault("decision_context", {})["dft_validation_feedback"] = {
        "error": error,
        "instruction": "保留记忆与当前证据，修订原方案；不要由代码替选。仍需人工重新批准。",
    }
    revision = revise(
        build_agent_proposal(action, decision_state),
        "本地核验未通过：" + error + "。请自行重新权衡并返回合规DFT方案，不执行。",
        state=revision_state,
        allowed_tools=["select_dft_candidates"],
        agent_client=agent_client,
        source_state=state,
        config=config,
    )
    if not isinstance(revision, dict):
        return DFTProposalPreparation(
            deepcopy(action), state, error + "；LLM修订返回无效结果，原方案保留，未执行任务。"
        )
    usage = revision.get("llm_usage") or (revision.get("action") or {}).get("_llm_usage")
    if usage:
        state = record_budget_usage(
            state, {"llm_usage": usage, "iteration": state.get("iteration", 0)}
        )
    if revision.get("revision_status") != "revised":
        return DFTProposalPreparation(
            deepcopy(action),
            state,
            error + "；LLM修订失败，原方案保留，未执行任务。" + str(revision.get("analysis") or ""),
        )
    revised_action = revision.get("action")
    if (
        not isinstance(revised_action, dict)
        or revised_action.get("tool") != "select_dft_candidates"
    ):
        return DFTProposalPreparation(
            deepcopy(action), state, error + "；LLM修订未返回有效DFT方案，原方案保留，未执行任务。"
        )
    prepared, error = attach_dft_preview(revised_action, state, config)
    if error:
        return DFTProposalPreparation(deepcopy(action), state, error)
    if prepared:
        prepared["parameters"]["plan_revision_notice"] = (
            "原方案未通过核验；LLM已修订，需确认新方案。"
        )
    return DFTProposalPreparation(prepared, state)

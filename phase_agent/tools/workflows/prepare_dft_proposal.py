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


def _preview_with_complete_feedback(action, state, config):
    from phase_agent.tools.workflows.dft_coverage_review import coverage_errors

    rows = (action.get("parameters") or {}).get("decisions") or []
    candidates = state.get("qbc_candidates") or []
    known = {row.get("candidate_id") for row in candidates}
    errors = []
    if isinstance(rows, list) and all(isinstance(row, dict) for row in rows):
        errors.extend(
            "unknown_candidate:" + str(row.get("candidate_id"))
            for row in rows
            if row.get("candidate_id") not in known
        )
        errors.extend(coverage_errors(action, candidates))
    prepared, preview_error = attach_dft_preview(action, state, config)
    if errors:
        detail = "DFT 身份或D/H审阅未通过：" + "; ".join(errors)
        return None, detail + ("；" + preview_error if preview_error else "")
    return prepared, preview_error


def prepare_dft_proposal(action, state, decision_state, config, *, agent_client=None, revise):
    from phase_agent.configuration.schema.python_environments import dft_environment_error

    environment_error = dft_environment_error(action, config)
    if environment_error:
        return DFTProposalPreparation(deepcopy(action), state, environment_error)
    prepared, error = _preview_with_complete_feedback(action, state, config)
    if not error or agent_client is None:
        return DFTProposalPreparation(prepared if not error else deepcopy(action), state, error)
    from phase_agent.tools.workflows.dft_coverage_review import coverage_evidence

    revision_state = deepcopy(decision_state)
    revision_state.setdefault("decision_context", {})["dft_validation_feedback"] = {
        "error": error,
        "prior_decisions": deepcopy((action.get("parameters") or {}).get("decisions") or []),
        "allowed_candidate_ids": [row["candidate_id"] for row in state.get("qbc_candidates") or []],
        "coverage_review": coverage_evidence(state.get("qbc_candidates") or []),
        "instruction": "返回完整决策表而非差量；逐条保留或明确修订原选择及暂缓，不因补一项漏掉另一项。candidate_id须逐字复制合法目录，不截断或猜测；程序不会模糊匹配。先同时修复所有身份和D/H缺口，再核对整批。对独立簇暂缓须权衡其联合H及条件依赖，同相代表不是替代证据。不要由代码替选；仍需人工重新批准。",
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
    prepared, error = _preview_with_complete_feedback(revised_action, state, config)
    if error:
        return DFTProposalPreparation(deepcopy(action), state, error)
    if prepared:
        prepared["parameters"]["plan_revision_notice"] = (
            "原方案未通过核验；LLM已修订，需确认新方案。"
        )
    return DFTProposalPreparation(prepared, state)

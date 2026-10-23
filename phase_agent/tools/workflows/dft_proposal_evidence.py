"""Bind DFT candidate, cost and independent audit evidence to the same current pool."""

from copy import deepcopy


def refresh_dft_evidence(state, decision_state, config):
    decision_state["qbc_candidates"] = deepcopy(state["qbc_candidates"])
    decision_state.setdefault("decision_context", {})["qbc_candidates"] = deepcopy(
        state["qbc_candidates"]
    )
    from phase_agent.decisions.qbc_selection.recommend_post_mc_dft import (
        recommend_post_mc_dft,
    )

    from phase_agent.tools.workflows.dft_coverage_review import coverage_evidence

    decision_state["decision_context"]["dft_DH_coverage_review"] = coverage_evidence(
        state["qbc_candidates"]
    )
    recommendation = recommend_post_mc_dft(state["qbc_candidates"], config, state)
    pool_ids = {row["candidate_id"] for row in state["qbc_candidates"]}
    audit = state.get("dft_audit_candidate_ids") or []
    # A still-valid frozen audit remains binding during human proposal revision.
    if not audit or not set(audit) <= pool_ids:
        audit = [
            r["candidate_id"]
            for r in recommendation["selected_candidates"]
            if r.get("dft_selection_mode") == "independent_audit"
        ]
    state["dft_audit_candidate_ids"] = list(audit)
    decision_state["dft_audit_candidate_ids"] = list(audit)
    decision_state["decision_context"]["independent_audit_candidate_ids"] = state[
        "dft_audit_candidate_ids"
    ]
    decision_state["decision_context"]["dft_sampling_recommendation"] = {
        "candidate_ids": [row["candidate_id"] for row in recommendation["selected_candidates"]],
        "summary": recommendation["summary"],
        "instruction": "此为可选参考而非最终决定；结合记忆与现状自行选择。兼顾near-hull、相、Na、已知QBC与抽查，缺失QBC不评分；偏离覆盖目标说明理由。",
    }
    from phase_agent.tools.budget.estimate_stage_cost import estimate_dft_cost

    decision_state["decision_context"]["dft_candidate_costs"] = [
        {
            "candidate_id": row["candidate_id"],
            "single_point_cost": estimate_dft_cost("DFT_SINGLE_POINT", row, config["qbc"]),
            "relax_cost": estimate_dft_cost("DFT_RELAX", row, config["qbc"]),
        }
        for row in state["qbc_candidates"]
    ]
    decision_state["decision_context"]["dft_stage_limits"] = (config.get("budgets") or {}).get(
        "stage_limits"
    )

"""Bind DFT candidate, cost and independent audit evidence to the same current pool."""

from copy import deepcopy


def refresh_dft_evidence(state, decision_state, config):
    decision_state["qbc_candidates"] = deepcopy(state["qbc_candidates"])
    decision_state.setdefault("decision_context", {})["qbc_candidates"] = deepcopy(
        state["qbc_candidates"]
    )
    import random
    from phase_agent.agents.reliability.selection.dft_round_policy import (
        dft_task_limit,
        round_dft_tasks,
    )

    from phase_agent.tools.workflows.dft_coverage_review import coverage_evidence

    decision_state["decision_context"]["dft_DH_coverage_review"] = coverage_evidence(
        state["qbc_candidates"]
    )
    pool_ids = {row["candidate_id"] for row in state["qbc_candidates"]}
    # Keep surviving members of the already registered independent draw.  A
    # filtered-out candidate must not silently replace the whole draw, changing
    # an approved selection's audit identity during proposal execution.
    audit = [cid for cid in (state.get("dft_audit_candidate_ids") or []) if cid in pool_ids]
    if not audit:
        done = {row.get("structure_id") for row in round_dft_tasks(state, config)}
        eligible = [
            row["candidate_id"]
            for row in state["qbc_candidates"]
            if row["candidate_id"] not in done
        ]
        random.Random(0).shuffle(eligible)
        limit = dft_task_limit(config)
        fraction = float((config.get("dft") or {}).get("independent_audit_fraction", 0.05))
        if not 0 <= fraction <= 1:
            raise ValueError("independent_audit_fraction must be in [0, 1]")
        count = min(len(eligible), limit, max(1, round(limit * fraction)) if fraction else 0)
        audit = eligible[:count]
    state["dft_audit_candidate_ids"] = list(audit)
    decision_state["dft_audit_candidate_ids"] = list(audit)
    decision_state["decision_context"]["independent_audit_candidate_ids"] = state[
        "dft_audit_candidate_ids"
    ]
    decision_state["decision_context"]["dft_sampling_recommendation"] = {
        "candidate_ids": [],
        "summary": {"status": "agent_decision_required"},
        "instruction": "不得使用固定加权或把上限当目标数量。Agent应根据完整基态、Ehull、可用QBC、D/H、覆盖和外推失败情况确定数量与选择；无同帧证据不得推断安全，另抽少量高Ehull点。",
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

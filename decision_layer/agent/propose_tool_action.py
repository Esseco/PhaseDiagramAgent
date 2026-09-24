"""Request one structured tool action, with deterministic rule fallback."""
from config_layer.schema.action_state_schema import normalize_action


def propose_agent_tool_action(state: dict, *, agent_client=None, allowed_tools: list[str]) -> dict:
    if agent_client is not None:
        try:
            decision_context = state.get("decision_context") or {}
            action = agent_client({"mode": "autonomous_search", "state": state, "decision_context": decision_context, "allowed_tools": allowed_tools, "instruction": "Choose one registered tool. Return tool, task_key, target_ids, parameters, budget, reason and evidence_refs. For allocate_mc_bohb, select a batch of existing branch_id values from decision_context.available_branches in target_ids; put focus_regions, exploration_fraction, mc_budget and dft_budget in parameters. Do not assign per-branch MC budgets: Hyperband alone controls fidelity and promotions after the selected branches are Relax-screened. For select_dft_candidates, read existing qbc_candidates only and put categorical decisions [{candidate_id, action, reason}] plus global_action in parameters; allowed actions are DFT_SINGLE_POINT, DFT_RELAX, DEFER, REJECT. DFT has no fixed quota: recommend zero or only a small representative set when MLIP final-structure predictions are already accurate; use DFT to calibrate/correct MLIP, not broad coverage. QBC supplies uncertainty only. Consult decision_context and its usage_rules. Do not invent energies, Ehull, scores, uncertainty values or convergence."})
            if not isinstance(action, dict):
                raise TypeError("agent action must be dict")
            return normalize_action({**action, "decision_source": "llm_agent", "decision_context": decision_context})
        except Exception as error:
            fallback_reason = f"llm_failed: {type(error).__name__}: {error}"
    else:
        fallback_reason = "llm_not_configured"
    tool = "check_convergence" if "check_convergence" in allowed_tools else "pause_search"
    return normalize_action({"tool": tool, "target_ids": [], "parameters": {}, "budget": 0.0, "reason": "rule fallback", "decision_source": "rule", "fallback_reason": fallback_reason})

"""Ask an LLM only for a structured round-level strategy."""

from phase_agent.decisions.strategy.choose_rule_round_strategy import choose_rule_round_strategy
from phase_agent.configuration.schema.action_state_schema import validate_state_schema


def propose_round_strategy(summary: dict, *, agent_client=None, config: dict) -> dict:
    if agent_client is None:
        return choose_rule_round_strategy(summary, config=config)
    errors = validate_state_schema(summary)
    if errors:
        raise ValueError("Round Agent requires StateSnapshot: " + ",".join(errors))
    payload = {
        "state": summary,
        "instruction": "Return only a round strategy: focus_regions, generation_quotas, mc_budget, dft_budget, exploration_fraction, reason. The Agent-selected Branch batch is already frozen; do not assign per-branch fidelity. Hyperband controls MC fidelity/promotions. Do not output branch scores, per-branch budgets, task IDs, retries, DFT structures, energies, Ehull, rewards, or convergence.",
    }
    try:
        payload["decision_context"] = summary.get("decision_context") or {}
        payload["instruction"] += (
            " Consult decision_context and its usage_rules; cite evidence in reason."
        )
        result = agent_client(payload)
        if not isinstance(result, dict):
            raise TypeError("LLM output must be a dict")
        return {**result, "source": "llm_agent", "decision_level": "round_strategy"}
    except Exception as error:
        fallback = choose_rule_round_strategy(summary, config=config)
        fallback["fallback_reason"] = f"llm_call_failed: {type(error).__name__}: {error}"
        return fallback

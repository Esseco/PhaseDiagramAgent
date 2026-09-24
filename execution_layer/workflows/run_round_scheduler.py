"""Coordinate one frozen LLM strategy round with BOHB task allocation."""

from copy import deepcopy

from decision_layer.strategy.choose_rule_round_strategy import choose_rule_round_strategy
from execution_layer.workflows.create_round_strategy_snapshot import create_round_strategy_snapshot
from decision_layer.strategy.propose_round_strategy import propose_round_strategy
from data_layer.ledger.register_effective_decisions import register_effective_decisions
from decision_layer.strategy.validate_round_strategy import validate_round_strategy
from scientific_layer.bohb.run_bohb_iteration import run_bohb_iteration
from execution_layer.state.state_manager import agent_state_summary, update_state_snapshot


def run_round_scheduler(state: dict | None, candidates: list[dict], *, summary: dict, bohb_config: dict, strategy_config: dict, invocation_id: str, agent_client=None, evaluator=None, proposed_strategy=None) -> dict:
    current = deepcopy(state or {"round_index": 0, "round_history": [], "active_round": None})
    active = current.get("active_round")
    if active and invocation_id in active.get("invocations", {}):
        return {**deepcopy(active["invocations"][invocation_id]), "state": current, "idempotent_replay": True}
    if active and active.get("status") != "active":
        return {"status": "round_not_active", "state": current, "actions": [], "round": active}
    if active is None:
        summary = {**current, **summary}
        snapshot_state = update_state_snapshot(summary)
        decision_snapshot = agent_state_summary(snapshot_state)
        proposed = deepcopy(proposed_strategy) if proposed_strategy is not None else propose_round_strategy(decision_snapshot, agent_client=agent_client, config=strategy_config)
        if proposed_strategy is not None:
            proposed.setdefault("source", "agent_tool_action")
            proposed.setdefault("decision_level", "round_strategy")
        validation = validate_round_strategy(proposed, summary, config=strategy_config)
        fallback_used = proposed.get("source") == "rule" and bool(proposed.get("fallback_reason"))
        if not validation["valid"]:
            reason = ",".join(validation["errors"])
            proposed = choose_rule_round_strategy(summary, config=strategy_config)
            proposed["fallback_reason"] = f"invalid_llm_strategy: {reason}"
            validation = validate_round_strategy(proposed, summary, config=strategy_config)
            fallback_used = True
        if not validation["valid"]:
            return {"status": "invalid_rule_fallback", "state": current, "actions": [], "validation": validation}
        active = create_round_strategy_snapshot(round_index=current["round_index"] + 1, strategy=proposed, candidates=candidates, bohb_config=bohb_config)
        if active.get("status") != "active":
            return {"status": "not_configured", "state": current, "actions": [], "round": active}
        active["strategy_validation"] = validation; active["fallback_used"] = fallback_used
        current["round_index"] += 1; current["active_round"] = active
    frozen_ids = active["candidate_ids"]
    if sorted(item["branch_id"] for item in candidates) != frozen_ids:
        return {"status": "candidate_pool_changed", "state": current, "actions": [], "required_action": "pause_or_end_round_then_create_new_version"}
    incoming_scope = create_round_strategy_snapshot(round_index=current["round_index"], strategy=active["strategy"], candidates=candidates, bohb_config=bohb_config)
    if incoming_scope.get("bohb_scope", {}).get("scope_id") != active["bohb_scope"]["scope_id"]:
        return {"status": "scope_changed", "state": current, "actions": [], "required_action": "pause_or_end_round_then_create_new_version"}
    output = run_bohb_iteration(candidates, active.get("bohb_state"), config=active["bohb_config"], total_mc_budget=int(active["strategy"]["mc_budget"]), seed=int(summary.get("seed", 0)) + int(active["round_id"].split("-")[-1]), evaluator=evaluator)
    active["bohb_state"] = output["state"]
    registered = register_effective_decisions(active, output["actions"])
    active = registered["round_state"]
    response = {"status": output["status"], "actions": registered["accepted"], "duplicate_decisions": registered["duplicates"], "strategy_version": active["strategy_version"], "decision_source": active["decision_source"], "dft_budget": active["strategy"]["dft_budget"], "dft_selection_owner": "decision_layer"}
    active.setdefault("invocations", {})[invocation_id] = deepcopy(response)
    current["active_round"] = active
    return {**response, "state": current, "round": active, "idempotent_replay": False}

"""Freeze the policy and BOHB scope at the start of a round."""

import hashlib
import json
from copy import deepcopy

from phase_agent.science.bohb.validate_bohb_scope import validate_bohb_scope


def create_round_strategy_snapshot(
    *, round_index: int, strategy: dict, candidates: list[dict], bohb_config: dict
) -> dict:
    effective_bohb_config = deepcopy(bohb_config)
    effective_bohb_config["agent_search_scope"] = {
        "focus_regions": list(strategy.get("focus_regions") or []),
        "exploration_fraction": float(strategy.get("exploration_fraction", 0.0)),
    }
    scope = validate_bohb_scope(effective_bohb_config, candidates)
    if scope.get("status") != "completed":
        return {"status": "not_configured", "scope": scope}
    payload = {
        "round_index": round_index,
        "strategy": strategy,
        "scope_id": scope["scope_id"],
        "candidate_ids": sorted(item["branch_id"] for item in candidates),
        "objective": effective_bohb_config.get("objective"),
        "budget_levels": effective_bohb_config.get("budget_levels"),
        "eta": effective_bohb_config.get("eta"),
    }
    version = (
        "strategy-"
        + hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]
    )
    return {
        "status": "active",
        "round_id": f"round-{round_index:06d}",
        "strategy_version": version,
        "decision_source": strategy.get("source"),
        "decision_reason": strategy.get("reason"),
        "strategy": deepcopy(strategy),
        "bohb_scope": scope,
        "candidate_ids": payload["candidate_ids"],
        "bohb_config": effective_bohb_config,
        "bohb_state": None,
        "effective_decisions": {},
        "invocations": {},
        "branch_selection_owner": "agent",
        "mc_fidelity_owner": "hyperband",
        "dft_selection_owner": "decision_layer",
        "legality_budget_status_owner": "execution_layer",
    }

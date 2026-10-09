"""Validate categorical DFT decisions and apply deterministic hard triggers."""

from execution_layer.budget.estimate_stage_cost import estimate_dft_cost
from execution_layer.budget.check_budget import check_budget
from copy import deepcopy
from decision_layer.agent.dft_contracts import dft_contract_errors


def validate_dft_agent_decisions(proposal: dict, metrics: list[dict], state: dict, *, config: dict, config_version: str, remaining_budget: float) -> dict:
    errors, accepted, rejected = [], [], []
    budget_state = deepcopy(state)
    if str(proposal.get("source") or "").startswith("llm_agent") and _contains_forbidden_numeric({"decisions": proposal.get("decisions"), "global_action": proposal.get("global_action"), "reason": proposal.get("reason")}):
        errors.append("agent_supplied_scientific_numeric_value")
    by_id = {item["candidate_id"]: item for item in metrics}; seen = set(); spent = 0.0
    decisions = proposal.get("decisions", [])
    if not isinstance(decisions, list) or any(not isinstance(item, dict) for item in decisions):
        return {"valid": False, "errors": ["invalid_dft_decisions_format"],
                "accepted": [], "rejected": [], "global_action": proposal.get("global_action")}
    contract_errors = dft_contract_errors(decisions)
    if contract_errors:
        if str(proposal.get("source") or "").startswith("llm_agent"):
            for item in decisions:
                extra = set(item) - {"candidate_id", "action", "reason"}
                if extra:
                    errors.append(f"agent_decision_fields_forbidden:{','.join(sorted(extra))}")
        return {"valid": False, "errors": errors + contract_errors, "accepted": [],
                "rejected": [], "global_action": proposal.get("global_action")}
    relax_limit = max(1, int(len(decisions) * float((config.get("selection_policy") or {}).get(
        "max_relax_fraction", 0.10)))) if decisions else 0
    relax_accepted = 0
    for item in decisions:
        extra_fields = set(item) - {"candidate_id", "action", "reason"}
        if str(proposal.get("source") or "").startswith("llm_agent") and extra_fields:
            errors.append(f"agent_decision_fields_forbidden:{','.join(sorted(extra_fields))}")
            continue
        candidate_id, action = item.get("candidate_id"), item.get("action")
        if candidate_id not in by_id:
            errors.append(f"unknown_candidate:{candidate_id}"); continue
        if candidate_id in seen:
            errors.append(f"duplicate_candidate_decision:{candidate_id}"); continue
        seen.add(candidate_id)
        if action not in set(config.get("allowed_actions") or []):
            errors.append(f"invalid_action:{action}"); continue
        metric = by_id[candidate_id]
        if action == "DFT_RELAX" and relax_accepted >= relax_limit:
            rejected.append({"candidate_id": candidate_id, "action": action,
                             "reason": "dft_relax_reserved_for_limited_geometry_checks"}); continue
        hard = _hard_action(metric, config)
        if metric.get("duplicate_of"):
            if action not in {"REJECT", "DEFER"}:
                rejected.append({"candidate_id": candidate_id, "action": action,
                                 "reason": "duplicate_safety_rule"}); continue
            hard = "duplicate_safety_rule"
        elif hard and action in {"DEFER", "REJECT"}:
            rejected.append({"candidate_id": candidate_id, "action": action,
                             "reason": "configured_uncertainty_rule_requires_revision", "required_action": hard}); continue
        task_key = f"{config_version}:{candidate_id}:{action}"
        if task_key in (state.get("effective_decisions") or {}):
            rejected.append({"candidate_id": candidate_id, "action": action, "reason": "duplicate_task_key"}); continue
        cost = estimate_dft_cost(action, metric, config)
        if spent + cost > remaining_budget:
            rejected.append({"candidate_id": candidate_id, "action": action, "reason": "dft_budget"}); continue
        spent += cost
        stage = {"DFT_SINGLE_POINT": "dft_single_point", "DFT_RELAX": "dft_relax"}.get(action)
        if stage and config.get("budget_limits"):
            check = check_budget(budget_state, {"stage": stage, "relative_cost": cost, "tasks": 1}, config["budget_limits"])
            if not check["allowed"]:
                spent -= cost
                rejected.append({"candidate_id": candidate_id, "action": action, "reason": "dft_budget", "details": check["reasons"]})
                continue
            budget_state.setdefault("budget_reservations", {})[task_key] = {"stage": stage, "reserved_cost": cost, "status": "reserved"}
        accepted.append({"candidate_id": candidate_id, "branch_id": metric.get("branch_id"), "action": action, "task_key": task_key, "relative_cost": cost, "reason": item.get("reason"), "hard_trigger": hard, "config_version": config_version})
        if action == "DFT_RELAX":
            relax_accepted += 1
    global_action = proposal.get("global_action", "CONTINUE_DATA_COLLECTION")
    if global_action not in {"RETRAIN_MLIP", "CONTINUE_DATA_COLLECTION"}:
        errors.append("invalid_global_action")
    return {"valid": not errors, "errors": errors, "accepted": accepted, "rejected": rejected, "reserved_cost": spent, "global_action": global_action, "source": proposal.get("source")}


def _hard_action(metric, config):
    rule = config.get("extreme_uncertainty") or {}
    for key in ("f_std_max", "energy_std"):
        threshold = rule.get(key); value = metric.get(key)
        if threshold is not None and value is not None and float(value) >= float(threshold):
            return rule.get("hard_action", "DFT_SINGLE_POINT")
    return None


def _contains_forbidden_numeric(value, path=""):
    forbidden = {"energy", "ehull", "qbc", "uncertainty", "f_std_max", "f_std_p95", "energy_std", "converged", "score"}
    if isinstance(value, dict):
        for key, child in value.items():
            if any(token in str(key).lower() for token in forbidden) and isinstance(child, (int, float)):
                return True
            if _contains_forbidden_numeric(child, f"{path}.{key}"):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_numeric(item, path) for item in value)
    return False

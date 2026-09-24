"""Canonical Agent action and state snapshot schema helpers."""
from copy import deepcopy


ACTION_REQUIRED_FIELDS = ("action_type", "parameters", "reason", "expected_cost")
STATE_REQUIRED_FIELDS = ("current_status", "available_budget", "uncertainty", "search_history")


def normalize_action(action: dict) -> dict:
    if not isinstance(action, dict):
        raise TypeError("action must be a dict")
    normalized = deepcopy(action)
    action_type = normalized.get("action_type") or normalized.get("tool")
    normalized["action_type"] = action_type
    normalized.setdefault("tool", action_type)  # handler compatibility
    normalized["parameters"] = deepcopy(normalized.get("parameters") or {})
    normalized["reason"] = normalized.get("reason") or "unspecified"
    normalized["expected_cost"] = deepcopy(
        normalized.get("expected_cost", normalized.get("budget", 0.0))
    )
    normalized.setdefault("budget", _numeric_cost(normalized["expected_cost"]))
    return normalized


def validate_action_schema(action: dict) -> list[str]:
    errors = [f"missing_action_field:{field}" for field in ACTION_REQUIRED_FIELDS
              if field not in action or action[field] is None]
    if not isinstance(action.get("action_type"), str) or not action.get("action_type"):
        errors.append("invalid_action_type")
    if not isinstance(action.get("parameters"), dict):
        errors.append("invalid_action_parameters")
    if not isinstance(action.get("reason"), str):
        errors.append("invalid_action_reason")
    return errors


def validate_state_schema(snapshot: dict) -> list[str]:
    errors = [f"missing_state_field:{field}" for field in STATE_REQUIRED_FIELDS
              if field not in snapshot]
    if snapshot.get("schema_version") != 1:
        errors.append("unsupported_state_schema_version")
    if not isinstance(snapshot.get("current_status"), str):
        errors.append("invalid_state_current_status")
    budget = snapshot.get("available_budget")
    if budget is not None and (isinstance(budget, bool) or not isinstance(budget, (int, float, dict))):
        errors.append("invalid_state_available_budget")
    if not isinstance(snapshot.get("uncertainty"), (list, dict)):
        errors.append("invalid_state_uncertainty")
    if not isinstance(snapshot.get("search_history"), list):
        errors.append("invalid_state_search_history")
    return errors


def _numeric_cost(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, dict):
        for key in ("value", "relative_cost", "total"):
            candidate = value.get(key)
            if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
                return float(candidate)
    return 0.0

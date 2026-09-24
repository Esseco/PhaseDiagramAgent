"""Keep numerical convergence distinct from budget exhaustion."""


def normalize_convergence_result(result: dict) -> dict:
    normalized = dict(result or {})
    if normalized.get("budget_exhausted") and not normalized.get("criteria_satisfied"):
        normalized["converged"] = False
        normalized["search_status"] = "budget_exhausted"
        normalized["reason"] = normalized.get("reason") or "budget exhausted before convergence criteria"
    elif normalized.get("criteria_satisfied"):
        normalized["converged"] = True
        normalized["search_status"] = "converged"
    else:
        normalized.setdefault("converged", False)
        normalized.setdefault("search_status", "continue")
    return normalized

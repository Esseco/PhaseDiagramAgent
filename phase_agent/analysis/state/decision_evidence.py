"""Copy every candidate metric without copying atom-level transport payloads."""

from copy import deepcopy

ATOM_FIELDS = frozenset(
    {
        "structure",
        "initial_structure",
        "relaxed_structure",
        "sites",
        "cart_coords",
        "frac_coords",
        "T",
        "tm_site_order",
        "forces",
        "trajectory",
        "raw_output",
        "stdout",
        "stderr",
    }
)


def candidate_evidence(value):
    if isinstance(value, dict):
        return {
            key: candidate_evidence(item) for key, item in value.items() if key not in ATOM_FIELDS
        }
    if isinstance(value, list):
        return [candidate_evidence(item) for item in value]
    if isinstance(value, tuple):
        return tuple(candidate_evidence(item) for item in value)
    return deepcopy(value)


def transport_copy(payload):
    """Project candidate pools before deepcopy, leaving approval parameters intact."""
    result = {}
    for key, value in payload.items():
        if key in {"state", "decision_context"} and isinstance(value, dict):
            result[key] = evidence_context(value)
        else:
            result[key] = deepcopy(value)
    return result


def evidence_context(context):
    result = {}
    for key, value in context.items():
        if key in {"available_branches", "branch_candidates", "qbc_candidates"}:
            result[key] = candidate_evidence(value)
        elif key == "decision_context" and isinstance(value, dict):
            result[key] = evidence_context(value)
        elif key == "budget_usage" and isinstance(value, dict):
            result[key] = {
                name: {
                    k: deepcopy(v)
                    for k, v in item.items()
                    if k in {"calls", "input_tokens", "output_tokens", "cost"}
                }
                if name == "llm" and isinstance(item, dict)
                else deepcopy(item)
                for name, item in value.items()
            }
        else:
            result[key] = deepcopy(value)
    return result

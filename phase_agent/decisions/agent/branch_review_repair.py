"""One bounded model correction for an invalid branch review; no tool dispatch."""

from copy import deepcopy

from phase_agent.decisions.agent.branch_pool_review import review_errors
from phase_agent.tools.policy.region_sampling import errors as region_errors


def repair_branch_review(client, payload, action):
    context = payload.get("decision_context") or {}
    if not context.get("branch_pool_review_required") or action.get("tool") != "generate_branches":
        return action

    def problems(candidate):
        return review_errors(candidate, context) + region_errors(candidate.get("parameters") or {})

    errors = problems(action)
    if not errors:
        return action
    request = deepcopy(payload)
    request.update(
        invalid_branch_proposal=deepcopy(action),
        validation_errors=errors,
        instruction=payload.get("instruction", "")
        + " Correct this invalid branch proposal once using the supplied evidence. "
        "The model must choose every group and reason, including any explicit default. "
        "For every evidence_refs field, select exact keys from output_contracts.branch_review_evidence.references (or decision_context.evidence_catalog.references). JSON paths and branch IDs are not evidence reference keys. Correct every reported invalid citation; do not invent or silently omit evidence. "
        "For each parameters.regions item, phase must be exactly one of O1, O3, P3, OP2 as a single string; "
        "do not omit it or use all/multiple phases/a list; split intended phases into separate regions within configured boundaries. "
        "Return the complete corrected action; no tool execution or approval is granted.",
    )
    corrected = client(request)
    if isinstance(corrected.get("action"), dict) and not corrected.get("tool"):
        envelope = corrected
        corrected = {
            **envelope["action"],
            **{k: v for k, v in envelope.items() if k.startswith("_")},
        }
    if corrected.get("tool") != "generate_branches":
        raise ValueError("branch_review_repair_must_not_change_the_tool")
    remaining = problems(corrected)
    if remaining:
        raise ValueError("branch_review_invalid_after_one_correction: " + "; ".join(remaining))
    return corrected

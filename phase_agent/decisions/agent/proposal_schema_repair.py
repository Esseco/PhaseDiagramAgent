"""One model correction of scientific wire contracts; never dispatch a tool."""

from copy import deepcopy

from phase_agent.decisions.agent.proposal_validation import (
    proposal_schema_errors,
    preserve_proposal_evidence,
)


def repair_proposal_schema(client, payload, action):
    context = payload.get("decision_context") or {}
    errors = proposal_schema_errors(action, context)
    if not errors:
        return action
    request = deepcopy(payload)
    request.update(
        mode="proposal_contract_repair",
        current_proposal=deepcopy(action),
        validation_errors=errors,
        instruction=payload.get("instruction", "")
        + " Correct the reported scientific action schema errors once using the supplied contracts. "
        "Return one complete action with exact candidate IDs, parameters and scientific reasons. "
        "The model must decide any correction; the program will not convert enums or choose candidates. "
        "All scientific, allowed-tool and approval checks still apply. No execution is authorized.",
    )
    corrected = client(request)
    if (
        isinstance(corrected, dict)
        and isinstance(corrected.get("action"), dict)
        and not (corrected.get("tool") or corrected.get("action_type"))
    ):
        envelope = corrected
        corrected = dict(envelope["action"])
        if envelope.get("_llm_usage"):
            corrected["_llm_usage"] = envelope["_llm_usage"]
    remaining = proposal_schema_errors(corrected, context)
    if remaining:
        error = ValueError(
            "Invalid LLM action: LangGraph proposal validation failed: " + "; ".join(remaining)
        )
        preserve_proposal_evidence(error, corrected)
        raise error
    return corrected

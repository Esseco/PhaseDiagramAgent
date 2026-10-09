"""LLM decision nodes; reuse scientific correction and evidence accounting."""
from langgraph.runtime import Runtime
from orchestration.runtime_context import ProposalRuntime


def _context(runtime):
    if not isinstance(runtime.context, ProposalRuntime):
        raise ValueError("Proposal graph requires context=ProposalRuntime(client, payload)")
    return runtime.context


def scientific_proposal(state, runtime: Runtime[ProposalRuntime]):
    from decision_layer.agent.post_dft_review import request_validated_action

    context = _context(runtime)
    return {"proposal": request_validated_action(context.agent_client, context.payload)}


def validate_proposal(state, runtime: Runtime[ProposalRuntime]):
    from decision_layer.agent.proposal_validation import proposal_errors, preserve_proposal_evidence

    action = state["proposal"]
    errors = proposal_errors(action, _context(runtime).payload)
    if errors:
        error = ValueError("Invalid LLM action: LangGraph proposal validation failed: " + "; ".join(errors))
        preserve_proposal_evidence(error, action)
        raise error
    return {}

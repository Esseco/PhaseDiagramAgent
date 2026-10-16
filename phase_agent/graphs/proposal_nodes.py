"""LLM decision nodes; reuse scientific correction and evidence accounting."""

from langgraph.runtime import Runtime
from phase_agent.graphs.runtime_context import ProposalRuntime


def _context(runtime):
    if not isinstance(runtime.context, ProposalRuntime):
        raise ValueError("Proposal graph requires context=ProposalRuntime(client, payload)")
    return runtime.context


def scientific_proposal(state, runtime: Runtime[ProposalRuntime]):
    from phase_agent.decisions.agent.post_dft_review import request_validated_action

    context = _context(runtime)
    from phase_agent.runtime.turn_process import process_event, proposal_summary

    process_event(
        "决策依据",
        {
            "允许的操作": context.payload.get("allowed_tools"),
            "本轮目标": context.payload.get("user_instruction")
            or context.payload.get("user_message"),
            "配置体系": ((context.payload.get("state") or {}).get("system") or {}).get("system_id"),
        },
    )
    action = request_validated_action(context.agent_client, context.payload)
    process_event("模型方案", proposal_summary(action))
    return {"proposal": action}


def validate_proposal(state, runtime: Runtime[ProposalRuntime]):
    from phase_agent.decisions.agent.proposal_validation import (
        proposal_errors,
        preserve_proposal_evidence,
    )

    action = state["proposal"]
    errors = proposal_errors(action, _context(runtime).payload)
    from phase_agent.runtime.turn_process import process_event, proposal_summary

    process_event("方案校验", {"通过": not errors, "具体错误": errors})
    if errors:
        error = ValueError(
            "Invalid LLM action: LangGraph proposal validation failed: " + "; ".join(errors)
        )
        preserve_proposal_evidence(error, action)
        raise error
    return {}

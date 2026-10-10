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
    if errors and (
        "action.tool: not allowed in this request" in errors
        or any(row.startswith("action.prerequisite:") for row in errors)
        or any(row.startswith("parameters.generation_plan:") for row in errors)
        or (
            _context(runtime).payload.get("unified_dialogue")
            and any(row.startswith("action.tool:") for row in errors)
        )
    ):
        from phase_agent.decisions.agent.post_dft_review import request_validated_action

        context = _context(runtime)
        repair_payload = {
            **context.payload,
            "mode": "proposal_contract_repair",
            "invalid_action": action,
            "validation_errors": errors,
            "instruction": context.payload.get("instruction", "")
            + " Repair the reported contract errors once using the original user request. In unified_dialogue use the supplied dialogue_outcome schema and its meaning: viewing settings is inspect_configuration; changing settings is configure; discussion is answer. None has tool or parameters. A requested scientific proposal is a tool action awaiting approval. Only a scientific action uses a tool from allowed_tools. Never invent authorization or data. Never return update_mlip when training is already completed.",
        }
        repaired = request_validated_action(context.agent_client, repair_payload)
        errors = proposal_errors(repaired, context.payload)
        process_event("修正方案", proposal_summary(repaired))
        process_event("修正后校验", {"通过": not errors, "具体错误": errors})
        if not errors:
            return {"proposal": repaired}
        action = repaired
    if errors:
        error = ValueError(
            "Invalid LLM action: LangGraph proposal validation failed: " + "; ".join(errors)
        )
        preserve_proposal_evidence(error, action)
        raise error
    return {}

"""Read-only role adapters; only the final proposal leaves this graph."""

from langgraph.runtime import Runtime
from phase_agent.graphs.decision_team.runtime import DecisionTeamRuntime


def prepare_evidence(state, runtime: Runtime[DecisionTeamRuntime]):
    from phase_agent.evidence.snapshot import evidence_snapshot

    if not runtime.context.team_enabled():
        return {}
    return {"snapshot": evidence_snapshot(runtime.context.payload), "advice": []}


def consultation_plan(state, runtime: Runtime[DecisionTeamRuntime]):
    from phase_agent.agents.strategic.agent import plan_consultation

    plan = plan_consultation(runtime.context.expert_call, state["snapshot"], state.get("proposal"))
    from phase_agent.runtime.turn_process import process_event

    process_event(
        "专家咨询范围",
        {"角色": plan["consult"], "问题": plan["question"], "snapshot_id": plan["snapshot_id"]},
    )
    return {"consultation": plan}


def exploration(state, runtime: Runtime[DecisionTeamRuntime]):
    from phase_agent.agents.exploration.agent import assess

    advice = assess(
        runtime.context.expert_call, state["snapshot"], state["consultation"]["question"]
    )
    from phase_agent.runtime.turn_process import process_event

    process_event("探索专家建议", advice)
    return {"advice": [*state.get("advice", []), advice]}


def reliability(state, runtime: Runtime[DecisionTeamRuntime]):
    from phase_agent.agents.reliability.agent import assess

    advice = assess(
        runtime.context.expert_call, state["snapshot"], state["consultation"]["question"]
    )
    from phase_agent.runtime.turn_process import process_event

    process_event("可靠性专家建议", advice)
    return {"advice": [*state.get("advice", []), advice]}


def scientific_proposal(state, runtime: Runtime[DecisionTeamRuntime]):
    from phase_agent.agents.strategic.agent import propose

    action = propose(runtime.context.proposal_call, runtime.context.payload)
    return {"proposal": action}


def coordinated_proposal(state, runtime: Runtime[DecisionTeamRuntime]):
    from phase_agent.agents.strategic.agent import propose
    from phase_agent.evidence.snapshot import evidence_snapshot

    if (
        evidence_snapshot(runtime.context.payload)["snapshot_id"]
        != state["snapshot"]["snapshot_id"]
    ):
        raise ValueError("Scientific evidence changed during expert consultation")
    action = state.get("proposal") if not state.get("advice") else None
    if action is None:
        action = propose(
            runtime.context.proposal_call,
            runtime.context.payload,
            snapshot=state["snapshot"],
            advice=state.get("advice", []),
        )
    action["_decision_team"] = {
        "snapshot_id": state["snapshot"]["snapshot_id"],
        "consultation": state["consultation"],
        "advice": state.get("advice", []),
    }
    return {"proposal": action}


def validate_proposal(state, runtime: Runtime[DecisionTeamRuntime]):
    from phase_agent.agents.common.proposal_validation import (
        proposal_errors,
        preserve_proposal_evidence,
    )

    errors = proposal_errors(state["proposal"], runtime.context.payload)
    from phase_agent.runtime.turn_process import process_event

    process_event("方案校验", {"通过": not errors, "具体错误": errors})
    if errors:
        error = ValueError(
            "Invalid LLM action: LangGraph proposal validation failed: " + "; ".join(errors)
        )
        error.validation_errors = list(errors)
        preserve_proposal_evidence(error, state["proposal"])
        raise error
    return {}

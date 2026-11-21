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
    facts = state["snapshot"].get("facts") or {}
    rounds = (
        (facts.get("decision_context") or {}).get("reliability_extrapolation_rounds")
        or (facts.get("state") or {}).get("reliability_extrapolation_rounds")
        or {}
    ).get("rounds") or []
    if (
        any(row.get("direct_sampling_required") is True for row in rounds)
        and "reliability" not in plan["consult"]
    ):
        plan = {**plan, "consult": [*plan["consult"], "reliability"]}
    elif (
        any(
            isinstance(row, dict) and row.get("status") == "accepted"
            for row in ((facts.get("state") or {}).get("remote_dft_sampling") or {}).values()
        )
        and not rounds
        and ((facts.get("decision_context") or {}).get("post_mc_hull_decision_facts") or {}).get(
            "unexplored_count", 0
        )
        == 0
        and "exploration" in plan["consult"]
    ):
        plan = {**plan, "consult": [role for role in plan["consult"] if role != "exploration"]}
    elif (facts.get("decision_context") or {}).get(
        "skip_reliability_before_first_mc"
    ) is True and "reliability" in plan["consult"]:
        plan = {**plan, "consult": [role for role in plan["consult"] if role != "reliability"]}
    from phase_agent.runtime.turn_process import process_event

    process_event(
        "专家咨询范围",
        {"角色": plan["consult"], "问题": plan["question"], "snapshot_id": plan["snapshot_id"]},
    )
    return {"consultation": plan}


def exploration(state, runtime: Runtime[DecisionTeamRuntime]):
    from concurrent.futures import ThreadPoolExecutor
    import contextvars
    from phase_agent.agents.exploration.agent import assess
    from phase_agent.agents.reliability.agent import assess as assess_reliability

    client = getattr(runtime.context.agent_client, "expert_client", runtime.context.agent_client)
    concurrent = (
        getattr(client, "supports_concurrent_experts", False) is True
        and "reliability" in state["consultation"]["consult"]
    )
    executor = None
    if concurrent:
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="expert-reliability")
        ctx = contextvars.copy_context()
        future = executor.submit(
            ctx.run,
            assess_reliability,
            runtime.context.expert_call,
            state["snapshot"],
            state["consultation"],
        )
        runtime.context._pending_reliability = (executor, future)
    try:
        advice = assess(runtime.context.expert_call, state["snapshot"], state["consultation"])
    except BaseException:
        if executor is not None:
            pending = runtime.context._pending_reliability
            runtime.context._pending_reliability = None
            if pending is not None:
                pending[1].cancel()
                pending[0].shutdown(wait=True)
        raise
    from phase_agent.runtime.turn_process import process_event

    process_event("探索专家建议", advice)
    return {"advice": [*state.get("advice", []), advice]}


def reliability(state, runtime: Runtime[DecisionTeamRuntime]):
    from phase_agent.agents.reliability.agent import assess

    pending = runtime.context._pending_reliability
    runtime.context._pending_reliability = None
    if pending is not None:
        executor, future = pending
        try:
            advice = future.result()
        finally:
            executor.shutdown(wait=True)
    else:
        advice = assess(runtime.context.expert_call, state["snapshot"], state["consultation"])
    from phase_agent.runtime.turn_process import process_event

    process_event("可靠性专家建议", advice)
    return {"advice": [*state.get("advice", []), advice]}


def scientific_proposal(state, runtime: Runtime[DecisionTeamRuntime]):
    from phase_agent.agents.strategic.agent import propose

    action = propose(runtime.context.proposal_call, runtime.context.payload)
    return {"proposal": action}


def detect_expert_conflict(advice_list: list[dict]) -> bool:
    """Trigger bounded rebuttal only when exploration and reliability directly oppose."""
    by_role = {
        row.get("role"): row.get("recommendation")
        for row in advice_list or []
        if isinstance(row, dict)
    }
    exp_rec = by_role.get("exploration")
    rel_rec = by_role.get("reliability")
    return exp_rec in {"explore", "deepen"} and rel_rec in {"finetune", "dft", "defer"}


def _compact_summary(item: dict) -> dict:
    return {
        "role": item.get("role"),
        "recommendation": item.get("recommendation"),
        "assessment": str(item.get("assessment") or "")[:180],
        "evidence_refs": list(item.get("evidence_refs") or [])[:8],
    }


def _run_conditional_rebuttal(client, snapshot: dict, advice_list: list[dict]) -> list[dict]:
    from phase_agent.agents.contracts import RebuttalResponse, validate_rebuttal

    summaries = [_compact_summary(item) for item in advice_list]
    schema = RebuttalResponse.model_json_schema()
    rebuttals = []
    for role in ("exploration", "reliability"):
        other = next((s for s in summaries if s["role"] != role), None)
        own = next((s for s in summaries if s["role"] == role), None)
        if not other or not own:
            continue
        request = {
            "mode": "expert_rebuttal",
            "agent_role": role,
            "snapshot_id": snapshot["snapshot_id"],
            "own_summary": own,
            "peer_summary": other,
            "allowed_tools": [],
            "instruction": "Respond concisely (<=200 chars) to the peer specialist's conflicting recommendation.",
            "output_contracts": {"rebuttal_response": schema},
        }
        rebuttals.append(validate_rebuttal(client(request), role=role, snapshot=snapshot))
    return rebuttals


def coordinated_proposal(state, runtime: Runtime[DecisionTeamRuntime]):
    from phase_agent.agents.strategic.agent import propose
    from phase_agent.evidence.snapshot import evidence_snapshot

    if (
        evidence_snapshot(runtime.context.payload)["snapshot_id"]
        != state["snapshot"]["snapshot_id"]
    ):
        raise ValueError("Scientific evidence changed during expert consultation")
    advice = list(state.get("advice", []))
    post_mc_orthogonal = bool(
        ((state["snapshot"].get("facts") or {}).get("decision_context") or {}).get(
            "post_mc_hull_decision_facts"
        )
    )
    rebuttals = (
        _run_conditional_rebuttal(runtime.context.expert_call, state["snapshot"], advice)
        if not post_mc_orthogonal and detect_expert_conflict(advice)
        else []
    )
    if rebuttals:
        by_role = {r["role"]: r for r in rebuttals}
        advice = [
            {**a, **({"rebuttal": by_role[a["role"]]} if a.get("role") in by_role else {})}
            for a in advice
        ]
    action = state.get("proposal") if not advice else None
    if action is None:
        action = propose(
            runtime.context.proposal_call,
            runtime.context.payload,
            snapshot=state["snapshot"],
            advice=advice,
        )
    action["_decision_team"] = {
        "snapshot_id": state["snapshot"]["snapshot_id"],
        "consultation": state["consultation"],
        "advice": advice,
        **({"rebuttals": rebuttals} if rebuttals else {}),
    }
    return {"proposal": action}


def validate_proposal(state, runtime: Runtime[DecisionTeamRuntime]):
    from phase_agent.agents.common.proposal_validation import (
        proposal_errors,
        preserve_proposal_evidence,
    )

    errors = proposal_errors(state["proposal"], runtime.context.payload)
    from phase_agent.agents.common.dialogue_contract import is_dialogue
    from phase_agent.agents.permissions import missing_role_reasons

    client, payload = runtime.context.agent_client, runtime.context.payload
    # Native/frozen adapters validate original bindings at their own entry. Never
    # call the project model to recreate their already reviewed expert opinions.
    if (
        not errors
        and not is_dialogue(state["proposal"])
        and payload.get("mode") == "autonomous_search"
        and not getattr(client, "_validates_original_facts", False)
        and not getattr(client, "_codex_structured", False)
        and getattr(client, "_decision_source", None) != "codex"
    ):
        missing = missing_role_reasons(
            state["proposal"],
            state.get("advice", []),
            snapshot_id=(state.get("snapshot") or {}).get("snapshot_id"),
            state=payload.get("state") or {},
            context=payload.get("decision_context") or {},
        )
        errors.extend(
            "missing_required_role:" + role + ": " + reason for role, reason in missing.items()
        )
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

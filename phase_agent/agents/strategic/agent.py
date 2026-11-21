"""Select consultation scope and produce one validated scientific proposal."""

from phase_agent.agents.contracts import ConsultationPlan, model_content
from phase_agent.agents.strategic.context import with_advice
from phase_agent.agents.strategic.prompts import CONSULTATION


def _consultation_snapshot(evidence_snapshot):
    """Keep identity, stage overview and blockers for consultation routing; omit heavy pools."""
    facts = evidence_snapshot.get("facts") or {}
    state = facts.get("state") or {}
    dc = facts.get("decision_context") or {}
    light_state = {
        k: state[k]
        for k in (
            "current_status",
            "active_model_version",
            "confirmed_config_version",
            "current_project",
            "task_counts",
            "post_recovery_next_step_pending",
        )
        if k in state
    }
    light_dc = {
        k: dc[k]
        for k in (
            "stage_snapshot",
            "scientific_overview",
            "new_model_generation",
            "round_budget_review_required",
            "branch_pool_review_required",
        )
        if k in dc
    }
    if isinstance(dc.get("mc_entry_conditions"), dict):
        light_dc["mc_entry_conditions"] = {
            k: dc["mc_entry_conditions"][k]
            for k in ("required", "ready", "status", "reason")
            if k in dc["mc_entry_conditions"]
        }
    if isinstance(dc.get("extrapolation_recovery"), dict):
        light_dc["extrapolation_recovery"] = {
            k: dc["extrapolation_recovery"][k]
            for k in ("status", "reason")
            if k in dc["extrapolation_recovery"]
        }
    if "post_dft_assessment" in dc:
        light_dc["post_dft_assessment"] = bool(dc["post_dft_assessment"])
    light_facts = {
        k: facts[k]
        for k in ("user_instruction", "user_message", "current_project", "allowed_tools")
        if k in facts
    }
    light_facts["state"] = light_state
    light_facts["decision_context"] = light_dc
    return {**evidence_snapshot, "facts": light_facts}


def plan_consultation(client, snapshot, draft=None):
    from phase_agent.evidence.snapshot import role_context

    ctx = role_context(snapshot, "strategic")
    ctx["evidence_snapshot"] = _consultation_snapshot(ctx["evidence_snapshot"])
    request = {
        "mode": "team_consultation",
        "agent_role": "strategic",
        **ctx,
        "allowed_tools": [],
        "instruction": CONSULTATION,
        "output_contracts": {"consultation_plan": ConsultationPlan.model_json_schema()},
    }
    if draft is not None:
        request["draft_proposal"] = draft
    plan = ConsultationPlan.model_validate(model_content(client(request)))
    if plan.snapshot_id != snapshot["snapshot_id"]:
        raise ValueError("Consultation plan refers to a stale evidence snapshot")
    return plan.model_dump()


def propose(client, payload, *, snapshot=None, advice=None):
    from phase_agent.agents.common.post_dft_review import request_validated_action

    if snapshot is not None:
        payload = with_advice(payload, snapshot, advice or [])
    from phase_agent.agents.strategic.research_plan import ResearchPlan, HumanReviewRequest

    payload = {
        **payload,
        "output_contracts": {
            **(payload.get("output_contracts") or {}),
            "research_plan": ResearchPlan.model_json_schema(),
            "human_review_request": HumanReviewRequest.model_json_schema(),
        },
    }
    return request_validated_action(client, payload)

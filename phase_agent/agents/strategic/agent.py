"""Select consultation scope and produce one validated scientific proposal."""

from phase_agent.agents.contracts import ConsultationPlan, model_content
from phase_agent.agents.strategic.context import with_advice
from phase_agent.agents.strategic.prompts import CONSULTATION


def plan_consultation(client, snapshot, draft=None):
    request = {
        "mode": "team_consultation",
        "agent_role": "strategic",
        "evidence_snapshot": snapshot,
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

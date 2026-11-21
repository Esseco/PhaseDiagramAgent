"""Shared bounded expert assessment runner with optional 1-step read-only drill-down."""

from __future__ import annotations

from phase_agent.agents.contracts import advice_schema, specialist_task, validate_advice
from phase_agent.evidence.drilldown import run_drilldown_queries


def run_expert_assessment(
    client,
    snapshot,
    question,
    *,
    role: str,
    instruction: str,
    context_builder,
) -> dict:
    """Run specialist assessment with at most one incremental read-only drill-down."""
    request = {
        **context_builder(snapshot),
        "mode": "expert_assessment",
        "agent_role": role,
        **specialist_task(question, role),
        "allowed_tools": [],
        "instruction": instruction,
        "output_contracts": {"expert_advice": advice_schema(role)},
    }
    first = validate_advice(client(request), role=role, snapshot=snapshot)
    queries = first.get("query_requests") or []
    if not queries:
        return first
    drilldown_results = run_drilldown_queries(snapshot, queries)
    followup_request = {
        "mode": "expert_drilldown",
        "agent_role": role,
        "snapshot_id": snapshot["snapshot_id"],
        "previous_advice": {
            "assessment": first.get("assessment"),
            "recommendation": first.get("recommendation"),
            "query_requests": queries,
        },
        "drilldown_results": drilldown_results,
        "allowed_tools": [],
        "instruction": instruction
        + " Refine your advice using drilldown_results; return query_requests=[].",
        "output_contracts": {"expert_advice": advice_schema(role)},
    }
    refined = validate_advice(client(followup_request), role=role, snapshot=snapshot)
    refined["drilldown_results"] = drilldown_results
    return refined

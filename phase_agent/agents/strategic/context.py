"""Opinions enrich a proposal request but never change its authority or scope."""

from copy import deepcopy
from phase_agent.agents.strategic.prompts import FINAL


def with_advice(payload, snapshot, advice):
    from phase_agent.tools.dispatch.scientific_operations import operation_readiness

    return {
        **deepcopy({key: value for key, value in payload.items() if key != "decision_context"}),
        "operation_readiness": operation_readiness(snapshot.get("facts") or {}),
        "agent_role": "strategic",
        "decision_context": {
            **deepcopy(payload.get("decision_context") or {}),
            "evidence_snapshot_id": snapshot["snapshot_id"],
            "expert_advice": deepcopy(advice),
        },
        "instruction": payload.get("instruction", "") + " " + FINAL,
    }

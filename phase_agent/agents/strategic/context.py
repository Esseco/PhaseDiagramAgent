"""Opinions enrich a proposal request but never change its authority or scope."""

from copy import deepcopy
from phase_agent.agents.strategic.prompts import FINAL


def with_advice(payload, snapshot, advice):
    return {
        **deepcopy(payload),
        "agent_role": "strategic",
        "decision_context": {
            **deepcopy(payload.get("decision_context") or {}),
            "evidence_snapshot_id": snapshot["snapshot_id"],
            "expert_advice": deepcopy(advice),
        },
        "instruction": payload.get("instruction", "") + " " + FINAL,
    }

"""Read-only model reliability reasoning over a frozen evidence snapshot."""

from phase_agent.agents.contracts import advice_schema, validate_advice
from phase_agent.agents.reliability.context import build_context
from phase_agent.agents.reliability.prompts import INSTRUCTION


def assess(client, snapshot, question):
    request = {
        **build_context(snapshot),
        "mode": "expert_assessment",
        "agent_role": "reliability",
        "question": question,
        "allowed_tools": [],
        "instruction": INSTRUCTION,
        "output_contracts": {"expert_advice": advice_schema("reliability")},
    }
    return validate_advice(client(request), role="reliability", snapshot=snapshot)

"""Read-only exploration reasoning over a frozen evidence snapshot."""

from phase_agent.agents.contracts import advice_schema, validate_advice
from phase_agent.agents.exploration.context import build_context
from phase_agent.agents.exploration.prompts import INSTRUCTION


def assess(client, snapshot, question):
    request = {
        **build_context(snapshot),
        "mode": "expert_assessment",
        "agent_role": "exploration",
        "question": question,
        "allowed_tools": [],
        "instruction": INSTRUCTION,
        "output_contracts": {"expert_advice": advice_schema("exploration")},
    }
    return validate_advice(client(request), role="exploration", snapshot=snapshot)

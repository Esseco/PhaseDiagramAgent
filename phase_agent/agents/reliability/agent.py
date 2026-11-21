"""Read-only model reliability reasoning over a frozen evidence snapshot."""

from phase_agent.agents.common.expert_runner import run_expert_assessment
from phase_agent.agents.reliability.context import build_context
from phase_agent.agents.reliability.prompts import INSTRUCTION


def assess(client, snapshot, question):
    return run_expert_assessment(
        client,
        snapshot,
        question,
        role="reliability",
        instruction=INSTRUCTION,
        context_builder=build_context,
    )

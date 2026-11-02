"""Shared constraints and exploration evidence for one consultation."""

from phase_agent.evidence.snapshot import role_context


def build_context(snapshot):
    return role_context(snapshot, "exploration")

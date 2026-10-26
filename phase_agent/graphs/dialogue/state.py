from dataclasses import dataclass
from typing import TypedDict


class TurnState(TypedDict, total=False):
    message: str
    history_restored: bool
    operation: str
    facts: dict
    verified_recovery: dict
    recovery_error: str
    recovery_just_approved: bool
    result: dict
    reply: str
    evidence: dict
    evidence_topics: list[str]
    evidence_reads: int


@dataclass
class TurnRuntime:
    handler: object
    messages: list
    conversation_id: str | None
    scientific_call: object = None

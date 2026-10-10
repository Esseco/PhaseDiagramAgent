from dataclasses import dataclass
from typing import TypedDict


class TurnState(TypedDict, total=False):
    message: str
    history_restored: bool
    operation: str
    facts: dict
    result: dict
    reply: str


@dataclass
class TurnRuntime:
    handler: object
    messages: list
    conversation_id: str | None
    scientific_call: object = None

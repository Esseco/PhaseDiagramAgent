"""Append a resumable dialogue message without modifying configuration."""

from copy import deepcopy


def record_config_dialogue(session: dict, *, role: str, message: str) -> dict:
    updated = deepcopy(session); updated.setdefault("dialogue", []).append({"type": "message", "role": role, "message": message})
    return updated

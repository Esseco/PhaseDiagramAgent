"""Serializable, invocation-local decision state; no scientific lifecycle migration."""

from typing import TypedDict


class DecisionTeamState(TypedDict, total=False):
    snapshot: dict
    consultation: dict
    advice: list[dict]
    proposal: dict

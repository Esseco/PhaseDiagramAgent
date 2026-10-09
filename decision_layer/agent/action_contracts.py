"""LLM action wire fields; schema validation is never execution permission."""

from typing import Annotated
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

Identifier = Annotated[str, StringConstraints(strict=True, min_length=1, pattern=r"\S")]


class ActionEnvelope(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    tool: Identifier
    parameters: dict = Field(default_factory=dict)
    target_ids: list[Identifier] = Field(default_factory=list)
    budget: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0
    reason: str | None = None
    expected_purpose: str | None = None
    task_key: Identifier | None = None
    evidence_refs: list[Identifier] = Field(default_factory=list)
    round_budget_review: dict | None = None


def action_contract_errors(action):
    if not isinstance(action, dict):
        return ["action must be an object"]
    errors = []
    if action.get("tool") and action.get("action_type") and action["tool"] != action["action_type"]:
        errors.append("action.tool: conflicts with action_type")
    fields = {key: action[key] for key in ActionEnvelope.model_fields if key in action}
    fields.setdefault("tool", action.get("action_type"))
    try:
        ActionEnvelope.model_validate(fields)
    except ValidationError as error:
        errors.extend("action." + ".".join(map(str, row["loc"])) + ": " + row["msg"]
                      for row in error.errors(include_input=False, include_url=False))
    return errors

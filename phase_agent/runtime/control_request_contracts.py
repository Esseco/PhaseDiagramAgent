"""Input shape validation, not approval or permission validation."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator
from phase_agent.runtime.control_ui_contracts import Identifier


class ControlRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    conversation_id: str = "local-control"


class ProposeRequest(ControlRequest):
    instruction: Identifier


class DecisionRequest(ControlRequest):
    decision: Literal["approve", "reject", "confirm_sensitive"]
    plan_id: Identifier
    state_version: Identifier
    proposal_hash: Identifier
    comment: str = ""


class PauseRequest(ControlRequest):
    reason: str = ""


class ConfigPatchRequest(ControlRequest):
    patch: dict[str, object]
    reasons: dict[str, str] | None = None
    impacts: dict | None = None

    @field_validator("patch")
    @classmethod
    def valid_paths(cls, patch):
        if any(
            not path or any(not part.strip() or part != part.strip() for part in path.split("."))
            for path in patch
        ):
            raise ValueError("patch paths must contain nonempty components")
        return patch


class ConfigConfirmRequest(ControlRequest):
    explicit: bool = False


class MemoryProposeRequest(ControlRequest):
    record: dict


class MemoryReviewRequest(ControlRequest):
    proposal_id: Identifier
    approved: bool = False


class SkillPublishRequest(ControlRequest):
    draft_directory: Identifier
    approved: bool = False
    version: Identifier = "1.0.0"


MODELS = {
    "/phase/propose": ProposeRequest,
    "/phase/decision": DecisionRequest,
    "/phase/pause": PauseRequest,
    "/phase/config/patch": ConfigPatchRequest,
    "/phase/config/confirm": ConfigConfirmRequest,
    "/phase/memory/propose": MemoryProposeRequest,
    "/phase/memory/review": MemoryReviewRequest,
    "/phase/memory/skills/import": ControlRequest,
    "/phase/memory/skills/publish": SkillPublishRequest,
}


def validate_control_request(path, body):
    if not isinstance(body, dict):
        raise ValueError("control body must be a JSON object")
    model = MODELS.get(path)
    if model:
        try:
            model.model_validate(body)
        except ValidationError as error:
            raise ValueError(
                "Invalid control request: "
                + "; ".join(
                    ".".join(map(str, row["loc"])) + ": " + row["type"]
                    for row in error.errors(include_input=False, include_url=False)
                )
            ) from None
    return body

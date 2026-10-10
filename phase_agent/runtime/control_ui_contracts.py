"""Read-only UI output contracts; approval authorization stays in policy code."""

from typing import Annotated
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

Count = Annotated[int, Field(ge=0)]
Identifier = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]


class StatusResponse(BaseModel):
    # Scientific summaries are still owned by their existing modules.
    model_config = ConfigDict(strict=True, extra="allow")
    summary_id: Identifier
    config_version: str | None
    model_version: str | None
    hull_version: str | None
    task_counts: dict[str, Count]
    task_stage_counts: dict[str, Count]
    job_counts: dict[str, Count]
    pending_approvals: Count
    memory_reviews: Count
    reserved_relative_cost: Annotated[float, Field(ge=0, allow_inf_nan=False)]


class PendingPlan(BaseModel):
    model_config = ConfigDict(strict=True, extra="allow")
    plan_id: Identifier
    invocation_id: Identifier
    revision: Count
    recommended_action: str | None
    target_ids: list[str]
    parameters: dict | None
    reason: str | None
    estimated_cost: float | dict | None
    config_version: str | None
    model_version: str | None
    state_version: Identifier
    proposal_hash: Identifier


class PendingResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    pending: list[PendingPlan]
    count: Count
    state_version: Identifier
    approval_url: str

    @model_validator(mode="after")
    def consistent_plans(self):
        if self.count != len(self.pending):
            raise ValueError("pending count mismatch")
        if len({row.plan_id for row in self.pending}) != len(self.pending):
            raise ValueError("duplicate pending plan")
        if any(
            row.state_version != self.state_version or row.plan_id != row.invocation_id
            for row in self.pending
        ):
            raise ValueError("pending plan identity or state version mismatch")
        return self


class TaskSummary(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    task_id: str | None
    task_key: str | None
    batch_id: str | None
    stage: str | None
    status: str | None
    model_version: str | None
    config_version: str | None


class TasksResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    tasks: list[TaskSummary]


class ConfigResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    status: str | None
    draft_revision: Count | None
    config: dict
    confirmed_snapshot: dict | None
    dialogue: list[dict]
    readiness: dict | None


class MemoryResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    active: dict
    records: list[dict]
    candidate_count: Count
    review_queue: list[dict]


def validate_control_response(model, response):
    try:
        model.model_validate(response)
    except ValidationError as error:
        raise ValueError(
            "Invalid UI response: "
            + "; ".join(
                ".".join(map(str, row["loc"])) + ": " + row["type"]
                for row in error.errors(include_input=False, include_url=False)
            )
        ) from None
    # Preserve extension data and old wire shape, not framework defaults.
    return response

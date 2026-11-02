"""Simulation reasoning describes execution; it never grants execution authority."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class TaskStep(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    kind: Literal["preflight", "invoke", "verify"]
    description: str = Field(min_length=1, max_length=1200)
    tool: str | None = None


class SimulationPlan(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    action_hash: str = Field(min_length=1)
    rationale: str = Field(min_length=1, max_length=2400)
    steps: list[TaskStep] = Field(min_length=3, max_length=8)
    checks: list[str] = Field(min_length=1, max_length=12)
    failure_policy: Literal["return_to_strategic"]


class SimulationDiagnosis(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    assessment: str = Field(min_length=1, max_length=2400)
    evidence: list[
        Literal[
            "execution.status",
            "execution.error",
            "execution.operation_outcome",
            "execution.result.status",
        ]
    ] = Field(min_length=1, max_length=4)
    recommendation: Literal["return_to_strategic", "request_human_review"]
    limitations: str = Field(min_length=1, max_length=1200)

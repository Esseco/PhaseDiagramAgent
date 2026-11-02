"""Structured research roadmaps; only the active existing operation is executable."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ResearchStep(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    step_id: str = Field(min_length=1, max_length=100, pattern=r"\S")
    operation: str = Field(min_length=1)
    priority: int = Field(ge=1)
    reason: str = Field(min_length=1, max_length=1200)
    depends_on: list[str] = Field(default_factory=list, max_length=8)
    estimated_cost: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class ResearchPlan(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    active_step_id: str = Field(min_length=1)
    steps: list[ResearchStep] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def bounded_dependencies(self):
        from phase_agent.tools.dispatch.scientific_operations import OPERATIONS

        seen = set()
        for step in self.steps:
            if step.operation not in OPERATIONS:
                raise ValueError("research_plan references an unknown registered operation")
            if step.step_id in seen or not set(step.depends_on).issubset(seen):
                raise ValueError("research_plan duplicate, forward or cyclic dependency")
            seen.add(step.step_id)
        if self.active_step_id != self.steps[0].step_id:
            raise ValueError(
                "research_plan must start with its current active step; future steps need fresh state and authorization"
            )
        return self


class HumanReviewRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    required: bool = False
    major_decision: bool = False
    expert_conflict: Literal["none", "minor", "major"] = "none"
    reason: str = Field(default="", max_length=2400)

    @model_validator(mode="after")
    def explain_review(self):
        if (
            self.required or self.major_decision or self.expert_conflict == "major"
        ) and not self.reason.strip():
            raise ValueError("human review requires a stated reason")
        return self

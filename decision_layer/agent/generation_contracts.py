"""Strict generation-allocation syntax, separate from budgets and science."""
from typing import Annotated, Literal
import math

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


PositiveInteger = Annotated[int, Field(strict=True, gt=0)]
GenerationStrategy = Literal["coverage", "composition", "competing_phase", "periodic_extension", "tm_ordering"]


class GenerationAllocation(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    strategy: GenerationStrategy
    quota: PositiveInteger
    phase: Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)]
    reason: Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=160)]
    na_min: float | None = None
    na_max: float | None = None
    max_det_H: PositiveInteger | None = None

    @model_validator(mode="after")
    def validate_na_range(self):
        lo, hi = self.na_min, self.na_max
        if lo is not None or hi is not None:
            if lo is None or hi is None or not math.isfinite(lo) or not math.isfinite(hi) or lo < 0 or hi < lo:
                raise ValueError("Na/O2范围必须有限且0≤下限≤上限，两端同时提供")
        return self

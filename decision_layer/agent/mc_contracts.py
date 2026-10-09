"""Common MC wire fields, independent of the scientific allocation algorithm."""

from typing import Annotated
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class MCAllocationFields(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    mc_budget: Annotated[int, Field(ge=0)]
    dft_budget: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0
    exploration_fraction: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] = 0.1
    seed: int = 0
    focus_regions: list[str] = Field(default_factory=list)


def mc_contract_errors(parameters, *, fallback_budget=0):
    """Validate common fields only; preserve existing internal preview parameters."""
    if not isinstance(parameters, dict):
        return ["MC parameters must be an object"]
    fields = {key: parameters[key] for key in MCAllocationFields.model_fields if key in parameters}
    fields.setdefault("mc_budget", fallback_budget)
    try:
        MCAllocationFields.model_validate(fields)
    except ValidationError as error:
        return ["parameters." + ".".join(map(str, row["loc"])) + ": " + row["msg"]
                for row in error.errors(include_input=False, include_url=False)]
    return []

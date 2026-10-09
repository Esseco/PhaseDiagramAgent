"""Wire format only; candidate identity and budgets remain execution checks."""

from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, StringConstraints, TypeAdapter, ValidationError


class DFTDecision(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    candidate_id: Annotated[str, StringConstraints(strict=True, min_length=1, pattern=r"\S")]
    action: Literal["DFT_SINGLE_POINT", "DFT_RELAX", "DEFER", "REJECT"]
    # Optional for existing stored and rule-generated proposals.
    reason: str | None = None


_DECISIONS = TypeAdapter(list[DFTDecision])


def dft_contract_errors(decisions):
    try:
        _DECISIONS.validate_python(decisions, strict=True)
    except ValidationError as error:
        return ["decisions." + ".".join(map(str, row["loc"])) + ": " + row["msg"]
                for row in error.errors(include_input=False, include_url=False)]
    return []

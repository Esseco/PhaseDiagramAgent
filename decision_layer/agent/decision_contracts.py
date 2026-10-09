"""Typed LLM output contracts; never execution authority or scientific evidence."""
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError


Assessment = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=240)]


class PostDFTReview(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    choice: Literal["search", "finetune", "revise_strategy", "supplement_dft", "convergence", "stop"]
    stop_status: Literal["continue", "scientifically_converged", "budget_stop", "blocked"]
    finetune_recommendation: Literal["now", "defer", "insufficient_evidence"]
    error_assessment: Assessment
    coverage_assessment: Assessment
    finetune_assessment: Assessment
    search_assessment: Assessment
    reference_assessment: Assessment
    round_findings: Assessment
    limitations: Assessment
    dft_assessment: Assessment
    convergence_assessment: Assessment


def review_contract_errors(value):
    try:
        PostDFTReview.model_validate(value)
    except ValidationError as error:
        return ["post_dft_review." + ".".join(map(str, row["loc"])) + ": " + row["msg"]
                for row in error.errors(include_input=False, include_url=False)]
    return []

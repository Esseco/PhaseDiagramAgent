"""Three scientific reasoning contracts; deterministic modules retain execution."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from phase_agent.decisions.agent.decision_contracts import Assessment


class Alternative(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    route: Assessment
    reason: Assessment


class SelectionPurpose(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    candidate_id: str = Field(min_length=1)
    purpose: Literal["hull_verification", "model_improvement", "coverage", "bias_check"]


class ScientificDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    point: Literal["branch_sampling", "dft_sampling", "round_direction"]
    config_version: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    bottleneck: Assessment
    rationale: Assessment
    alternatives: list[Alternative] = Field(min_length=1, max_length=3)
    expected_outcome: Assessment
    success_criterion: Assessment
    limitations: Assessment
    selection_purposes: list[SelectionPurpose] = Field(default_factory=list)
    module_execution: Literal["bundled", "stepwise"] | None = None


def decision_point(action, context):
    tool = action.get("tool") or action.get("action_type")
    if context.get("post_dft_assessment") and tool in {
        "generate_branches",
        "select_dft_candidates",
        "update_mlip",
        "check_convergence",
        "pause_search",
        "adjust_strategy",
    }:
        return "round_direction"
    if tool == "generate_branches":
        return "branch_sampling"
    if tool == "select_dft_candidates":
        return "dft_sampling"
    if tool in {"update_mlip", "check_convergence", "pause_search"}:
        return "round_direction"
    return None


def scientific_decision_errors(action, brief, context=None):
    if not isinstance(action, dict):
        return ["scientific_decision_action_must_be_object"]
    point = decision_point(action, context or {})
    if point is None:
        return []
    try:
        review = ScientificDecision.model_validate(action.get("scientific_decision"))
    except ValidationError as error:
        return ["scientific_decision:" + str(error)]
    errors = []
    if review.point != point:
        errors.append("scientific_decision_point_mismatch")
    for key in ("config_version", "model_version"):
        if getattr(review, key) != brief.get(key):
            errors.append("scientific_decision_stale_" + key)
    if (action.get("tool") or action.get("action_type")) == "select_dft_candidates":
        selections = (action.get("parameters") or {}).get("decisions") or []
        if not isinstance(selections, list) or any(not isinstance(r, dict) for r in selections):
            return errors + ["dft_decisions_must_be_object_list"]
        ids = [
            r.get("candidate_id")
            for r in selections
            if r.get("action") in {"DFT_SINGLE_POINT", "DFT_RELAX"}
        ]
        purposes = [r.candidate_id for r in review.selection_purposes]
        if len(purposes) != len(set(purposes)) or set(purposes) != set(ids):
            errors.append("dft_selection_purposes_must_match_selected_candidates")
    elif review.selection_purposes:
        errors.append("selection_purposes_require_dft_selection")
    return errors


INSTRUCTION = """Scientific decisions have THREE points. Branch: choose phase/composition/supercell
regions and quotas from coverage, prior gains, duplicate rate and downstream cost;
compare coverage, low-energy follow-up and expansion. Respect disabled strategies.
DFT: choose a complementary batch for hull verification, model improvement, coverage
and possible committee shared-bias checks. QBC alone is not truth. Avoid redundant
same-branch/environment points; diversity is evidence-based, not a rigid quota.
Direction: diagnose the bottleneck and compare training/model update, existing-pool
DFT, further branch search and convergence. Separate scientific convergence from
budget stop or blocked execution. Aggregate MAE alone cannot prove hull ordering.
Include scientific_decision matching the schema and exact current model/config,
one feasible alternative (or explain why unavailable), expected outcome, observable
success criterion and limitations. These are hypotheses, never measured facts.
For DFT assign a purpose to every selected candidate. Do not invent metrics.
Set module_execution only for an explicit user bundled/stepwise preference; otherwise omit it.
Be concise: each assessment is one short sentence, preferably under 80 Chinese characters.
Do not additionally output decision_review. reason and expected_purpose are brief action
labels, not another copy of the rationale. Omit irrelevant optional fields and empty lists.
When post_dft_review is required, put detailed route comparisons there; scientific_decision
contains only the concise bottleneck/conclusion, expectation and success criterion.
Never repeat input evidence tables, raw metrics or candidate lists inside explanations.
Modules perform Relax-MC allocation, stopping, recovery and hull updates. In default
bundled mode do not propose standalone MC allocation or intermediate input approval.
Explicit user maintenance and existing pending legacy work retain their original path.
Return one scientific choice, not a multistep future workflow. A new
Branch proposal includes a displayed cost-bounded module envelope when supported;
registered branches are diagnosed after generation by code, not an additional LLM call.
"""


def attach_decision_contract(payload):
    brief = (payload.get("decision_context") or {}).get("scientific_decision_brief") or {}
    if (brief.get("policy") or {}).get("schema_version") != 1:
        return payload
    context = dict(payload.get("decision_context") or {})
    context.pop("decision_review_format", None)  # The scientific contract already owns explanation.
    return {
        **payload,
        "decision_context": context,
        "output_contracts": {
            **(payload.get("output_contracts") or {}),
            "scientific_decision": ScientificDecision.model_json_schema(),
        },
        "instruction": payload.get("instruction", "") + " " + INSTRUCTION,
    }

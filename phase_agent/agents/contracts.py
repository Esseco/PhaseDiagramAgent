"""Read-only scientific opinions; none of these contracts grant execution authority."""

from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator

Role = Literal["exploration", "reliability"]
TaskText = Annotated[str, Field(min_length=1, max_length=1000)]


class SpecialistTask(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    question: str = Field(min_length=1, max_length=1000)
    completion_criteria: list[TaskText] = Field(min_length=1, max_length=4)


class ConsultationPlan(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    snapshot_id: str = Field(min_length=1)
    consult: list[Role] = Field(max_length=2)
    question: str = Field(min_length=1, max_length=1000)
    role_tasks: dict[Role, "SpecialistTask"] = Field(default_factory=dict)

    @field_validator("role_tasks")
    @classmethod
    def requested_tasks(cls, value, info):
        if set(value) - set(info.data.get("consult", [])):
            raise ValueError("Task supplied for an unrequested specialist")
        return value

    @field_validator("consult")
    @classmethod
    def unique_roles(cls, value):
        if len(value) != len(set(value)):
            raise ValueError("Each expert may be consulted at most once")
        return value


class ExpertOption(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    tool: str = Field(min_length=1)
    reason: str = Field(min_length=1, max_length=1000)
    target_ids: list[str] = Field(default_factory=list, max_length=1000)
    parameters: dict = Field(default_factory=dict)
    operation: str | None = Field(default=None, min_length=1)
    estimated_cost: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    expected_benefit: str | None = Field(default=None, min_length=1, max_length=1200)
    evidence_refs: list[str] = Field(default_factory=list, max_length=32)
    assumptions: list[TaskText] = Field(default_factory=list, max_length=6)
    counterevidence_refs: list[str] = Field(default_factory=list, max_length=32)
    cost_unit: str | None = Field(default=None, min_length=1, max_length=100)


class ExpertAdvice(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    role: Role
    snapshot_id: str = Field(min_length=1)
    assessment: str = Field(min_length=1, max_length=1200)
    recommendation: Literal[
        "explore", "deepen", "dft", "finetune", "validate", "defer", "insufficient_evidence"
    ]
    options: list[ExpertOption] = Field(max_length=3)
    evidence_refs: list[str] = Field(max_length=64)
    expected_outcome: str = Field(min_length=1, max_length=1000)
    limitations: str = Field(min_length=1, max_length=1200)
    risk_structure_ids: list[str] = Field(default_factory=list, max_length=1000)
    evidence_gaps: list[TaskText] = Field(default_factory=list, max_length=6)
    handoff_roles: list[Role] = Field(default_factory=list, max_length=2)
    query_requests: list[
        Literal["phase_error_breakdown", "sampling_weight_stats", "extrapolation_hotspots"]
    ] = Field(default_factory=list, max_length=2)


class RebuttalResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    role: Role
    snapshot_id: str = Field(min_length=1)
    recommendation: Literal[
        "explore", "deepen", "dft", "finetune", "validate", "defer", "insufficient_evidence"
    ]
    rebuttal: str = Field(min_length=1, max_length=400)
    evidence_refs: list[str] = Field(default_factory=list, max_length=16)


def validate_rebuttal(value, *, role, snapshot):
    item = RebuttalResponse.model_validate(model_content(value))
    if item.role != role or item.snapshot_id != snapshot["snapshot_id"]:
        raise ValueError("Rebuttal role or evidence snapshot does not match this consultation")
    unknown = set(item.evidence_refs) - set(snapshot["available_evidence_refs"])
    if unknown:
        raise ValueError(
            "Expert cited unavailable evidence references: " + ", ".join(sorted(unknown))
        )
    from phase_agent.agents.permissions import RECOMMENDATIONS

    if item.recommendation not in RECOMMENDATIONS[role]:
        raise ValueError("recommendation is outside specialist role permission")
    return item.model_dump()


def model_content(value):
    if not isinstance(value, dict):
        raise ValueError("Role response must be a JSON object")
    return {
        k: v
        for k, v in value.items()
        if k not in {"_llm_usage", "_proposal_harness", "_analysis_route"}
    }


def validate_advice(value, *, role, snapshot, additional_identifiers=()):
    advice = ExpertAdvice.model_validate(model_content(value))
    if advice.role != role or advice.snapshot_id != snapshot["snapshot_id"]:
        raise ValueError("Expert role or evidence snapshot does not match this consultation")
    unknown = set(advice.evidence_refs) - set(snapshot["available_evidence_refs"])
    for option in advice.options:
        unknown.update(
            set(option.evidence_refs + option.counterevidence_refs)
            - set(snapshot["available_evidence_refs"])
        )
    if unknown:
        raise ValueError(
            "Expert cited unavailable evidence references: " + ", ".join(sorted(unknown))
        )
    if any(option.tool not in snapshot["allowed_tools"] for option in advice.options):
        raise ValueError("Expert option is outside the current allowed tool scope")
    from phase_agent.agents.permissions import RECOMMENDATIONS, advisory_option_errors

    if advice.recommendation not in RECOMMENDATIONS[role]:
        raise ValueError("recommendation is outside specialist role permission")
    for option in advice.options:
        errors = advisory_option_errors(role, option)
        if errors:
            raise ValueError("; ".join(errors))
    if advice.risk_structure_ids and role != "reliability":
        raise ValueError("risk structure assessment belongs to reliability role")
    facts = snapshot.get("facts") or {}
    operations = [
        *(facts.get("scientific_operations") or []),
        *(facts.get("module_operations") or []),
    ]
    has_catalog = "scientific_operations" in facts or "module_operations" in facts
    identifiers = set()

    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {
                    "branch_id",
                    "candidate_id",
                    "structure_id",
                    "task_id",
                    "batch_id",
                    "model_id",
                } and isinstance(item, str):
                    identifiers.add(item)
                if key in {"candidate_ids", "branch_ids", "target_ids"} and isinstance(item, list):
                    identifiers.update(v for v in item if isinstance(v, str))
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(facts)
    identifiers.update(additional_identifiers)
    if set(advice.risk_structure_ids) - identifiers:
        raise ValueError("Expert cites unavailable risk structure IDs")
    for option in advice.options:
        if (
            has_catalog
            and option.operation is not None
            and option.operation not in {row["operation"] for row in operations}
        ):
            raise ValueError("Expert operation is outside the current snapshot catalog scope")
        if (
            has_catalog
            and option.operation is None
            and option.tool not in {"pause_search", "adjust_strategy"}
        ):
            visible = [
                row
                for row in operations
                if row["tool"] == option.tool
                and not any(
                    key in option.parameters and option.parameters[key] != value
                    for key, value in (row.get("parameters") or {}).items()
                )
            ]
            if option.tool == "update_mlip":
                visible = [
                    row
                    for row in visible
                    if (row.get("parameters") or {}).get("action", "RETRAIN_MLIP")
                    == option.parameters.get("action", "RETRAIN_MLIP")
                ]
            if not visible:
                raise ValueError(
                    "Expert tool proposal is outside the current snapshot catalog scope"
                )
        if set(option.target_ids) - identifiers:
            raise ValueError("Expert option cites unavailable target IDs")
        if (
            operations
            and "mode" in option.parameters
            and not any(
                row["tool"] == option.tool
                and (row.get("parameters") or {}).get("mode") == option.parameters["mode"]
                for row in operations
            )
        ):
            raise ValueError("Expert option is outside the registered operation scope")
    result = advice.model_dump()
    if not result["risk_structure_ids"]:
        result.pop("risk_structure_ids")
    if not result["evidence_gaps"]:
        result.pop("evidence_gaps")
    if not result["handoff_roles"]:
        result.pop("handoff_roles")
    for option in result["options"]:
        for key in ("evidence_refs", "counterevidence_refs", "assumptions", "cost_unit"):
            if not option.get(key):
                option.pop(key, None)
        for key in ("operation", "estimated_cost", "expected_benefit"):
            if option.get(key) is None:
                option.pop(key, None)
    return result


def advice_schema(role):
    from phase_agent.agents.permissions import RECOMMENDATIONS, ADVISORY_OPERATIONS

    schema = ExpertAdvice.model_json_schema()
    schema["properties"]["recommendation"]["enum"] = list(RECOMMENDATIONS[role])
    schema["$defs"]["ExpertOption"]["properties"]["operation"] = {
        "anyOf": [{"type": "string", "enum": sorted(ADVISORY_OPERATIONS[role])}, {"type": "null"}],
        "default": None,
    }
    return schema


def specialist_task(plan, role):
    """Shared goal, separate question; opinions never confer execution permission."""
    defaults = {
        "exploration": (
            "Assess exploration coverage and the marginal value of further search.",
            ["Identify coverage gaps and stopping uncertainty; do not judge model validity."],
        ),
        "reliability": (
            "Assess model and phase-stability evidence and its principal failure modes.",
            ["Identify trust limits and counterevidence; do not choose the global route."],
        ),
    }
    question, criteria = defaults[role]
    goal = plan if isinstance(plan, str) else plan["question"]
    custom = {} if isinstance(plan, str) else (plan.get("role_tasks") or {}).get(role, {})
    return {
        "decision_goal": goal,
        "question": custom.get("question", question),
        "completion_criteria": custom.get("completion_criteria", criteria),
        "authority": "read_only_advice; no execution, approval or global route selection",
        "report_requirements": "State conclusions, option-specific evidence_refs, assumptions, counterevidence_refs, cost_unit and unknowns. Declare evidence_gaps when insufficient and handoff_roles for cross-domain risks; never substitute for another role or invent supporting facts or perform checks.",
    }

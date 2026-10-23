"""Read-only stage facts and model-selected analysis scope, never action routing."""

from collections import Counter, defaultdict
from copy import deepcopy
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from phase_agent.decisions.agent.dialogue_contract import EvidenceRequestOutcome


class AnalysisPlan(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    stage: str = Field(min_length=1, max_length=240)
    question: str = Field(min_length=1, max_length=480)
    reasoning: Literal["routine", "scientific"]
    analysis_kind: Literal["verification", "input_preparation", "scientific_comparison"] = (
        "scientific_comparison"
    )
    comparison_tools: list[str] = Field(default_factory=list, max_length=32)
    comparison_operations: list[str] = Field(default_factory=list, max_length=32)
    reasoning_effort: Literal["low", "high", "max"] = "low"
    reasoning_basis: str | None = Field(default=None, max_length=480)

    def needs_reasoning(self):
        """A reasoning floor for declared choices; never select or ban a route."""
        scientific_tools = {
            "generate_branches",
            "select_dft_candidates",
            "update_mlip",
            "check_convergence",
            "adjust_strategy",
            "allocate_mc_bohb",
            "prepare_mlip_training",
            "activate_mlip_candidate",
            "reevaluate_candidates",
        }
        return (
            self.reasoning == "scientific"
            or self.analysis_kind == "scientific_comparison"
            or len(set(self.comparison_tools)) > 1
            or bool(scientific_tools.intersection(self.comparison_tools))
            or bool(
                set(self.comparison_operations).intersection(
                    {
                        "generate_branches",
                        "allocate_mlip_mc",
                        "select_dft_candidates",
                        "prepare_ordering_search",
                        "prepare_relax_mc_batch",
                        "prepare_mlip_training",
                        "activate_mlip_candidate",
                        "check_convergence",
                    }
                )
            )
        )


class ScientificEvidenceRequest(EvidenceRequestOutcome):
    view: Literal["overview", "full"] = "full"
    full_for: list[str] | None = Field(
        default=None,
        max_length=32,
        description="Tools whose complete prerequisites are needed NOW, independent of other considered routes. Explicit view without full_for preloads only requested topics/fields; legacy requests omitting both retain comparison-wide preload. This never restricts final tools.",
    )
    fields: list[str] | None = Field(default=None, min_length=1, max_length=128)
    analysis_plan: AnalysisPlan | None = None


# These embed execution/state snapshots, not a new scientific observation. They
# remain indexed and readable in full by exact field name, with revision binding.
ON_DEMAND_HISTORY = frozenset(
    {
        "event_history",
        "invocations",
        "action_records",
        "agent_state_summary",
    }
)


# Supporting explanations and packaging traces are available in full by exact field.
# Current measured pools, costs, accepted outcomes and unknown scientific fields stay default.
ON_DEMAND_SUPPORT = frozenset(
    {
        "decision_feedback",
        "recent_experience",
        "recent_generation_allocations",
        "relax_mc_batches",
    }
)


def stage_snapshot(sources):
    """Keep overlapping/versioned facts; the model interprets the current stage."""
    model = sources.get("active_model_version") or (sources.get("mlip_status") or {}).get(
        "active_version"
    )
    counts = defaultdict(Counter)
    tasks = sources.get("tasks") or []
    if isinstance(tasks, dict):
        tasks = tasks.values()
    for task in tasks:
        if isinstance(task, dict):
            counts[str(task.get("model_version") or "unknown")][
                str(task.get("status") or "unknown")
            ] += 1
    result = {
        "active_model_version": model,
        "config_version": sources.get("confirmed_config_version") or sources.get("config_version"),
        "recorded_status": deepcopy(sources.get("current_status")),
        "task_counts_by_model": {key: dict(value) for key, value in counts.items()},
        "pending_approval_count": len(sources.get("pending_execution_policies") or {}),
    }
    for name, keys in {
        "new_model_generation": (
            "required",
            "post_refresh",
            "ready",
            "model_version",
            "new_branch_count",
            "first_search_completed",
        ),
        "model_refresh": ("status", "old_model_version", "new_model_version", "wave"),
        "training_validation_planning": ("status", "model_version", "candidate_model_version"),
        "post_dft_assessment": ("status", "model_version", "config_version"),
        "latest_convergence_check": ("status", "model_version", "config_version"),
    }.items():
        value = sources.get(name)
        if isinstance(value, dict) and value:
            fact = {key: deepcopy(value[key]) for key in keys if key in value}
            version = value.get("model_version") or value.get("new_model_version")
            fact["available"] = True
            fact["matches_active_model"] = version == model if version and model else None
            result[name] = fact
    generation = sources.get("new_model_generation") or {}
    version_matches = bool(model and generation.get("model_version") == model)
    count = generation.get("new_branch_count")
    registration_known = (
        version_matches and type(count) is int and isinstance(generation.get("ready"), bool)
    )
    result["branch_milestones"] = {
        "model_version": generation.get("model_version"),
        "matches_active_model": version_matches,
        "registration_completed": bool(generation["ready"] and count > 0)
        if registration_known
        else None,
        "registered_branch_count": count if version_matches else None,
        "first_search_completed": generation.get("first_search_completed")
        if version_matches
        else None,
        "meaning": (
            "Branch generation/registration and subsequent Relax+MC search are DIFFERENT "
            "milestones. ready=true plus a positive new_branch_count for the current model "
            "records that branches ALREADY exist. first_search_completed=false only says "
            "their first search is not recorded complete; it does NOT mean generation is "
            "incomplete. required describes the post-refresh prerequisite policy, not an "
            "unsatisfied step. This is progress evidence, not a recommendation: additional "
            "branches, existing-pool search, DFT or other legal routes remain scientific choices."
        ),
    }
    if not tasks:
        grouped = defaultdict(lambda: defaultdict(Counter))
        for group in (sources.get("task_round_evidence") or {}).get("groups") or []:
            version = str((group.get("lineage") or {}).get("model_version") or "unknown")
            for row in group.get("stage_evidence") or []:
                for status, count in (row.get("status_counts") or {}).items():
                    if type(count) is int and count >= 0:
                        grouped[version][str(row.get("stage") or "unknown")][status] += count
        result["task_stage_counts_by_model"] = {
            version: {stage: dict(values) for stage, values in stages.items()}
            for version, stages in grouped.items()
        }
        result["task_stage_counts_source"] = (
            "recorded task_round_evidence lineage; missing identity remains unknown"
        )
    result["interpretation"] = (
        "Recorded facts may overlap or refer to older models; unknown identity is not a match. "
        "Determine the current question from these facts and the user request. No stage label "
        "selects or bans a scientific route, proves convergence, or grants approval."
    )
    return result


PLANNING_INSTRUCTION = (
    "First interpret stage_snapshot, current batch_return_facts and the CURRENT user request. "
    "Distinguish a factual question about existing registered work from a NEW scientific choice. "
    "If the unresolved question is current task/return verification, inspect that evidence first; "
    "do not reopen every strategy merely because the user says continue. Reopen scientific "
    "alternatives whenever new facts or the scientific question warrant it. State analysis_plan with "
    "stage, the unresolved question, analysis_kind and reasoning. First distinguish branch "
    "registration from search completion using branch_milestones; required=true is not proof "
    "that a prerequisite is unsatisfied. Set analysis_kind=verification only for verifying "
    "recorded facts, input_preparation for parameterizing an established scope, and "
    "scientific_comparison for choosing regions, candidates, allocations, routes, training or "
    "convergence. A task named 'prepare' may still require scientific comparison. Also "
    "choose the thinking tier independently of evidence size. Start bounded scientific "
    "comparisons of already computed evidence at reasoning_effort=low (thinking stays enabled). "
    "Use high when you can identify a concrete unresolved scientific conflict, methodological "
    "problem or new hypothesis requiring deeper reasoning; max is for exceptional complexity. "
    "State that specific need in reasoning_basis, rather than a generic need to compare routes. "
    "Many candidates, several tools, a large history, or a request to continue do not alone "
    "justify high effort. Genuine verification/input preparation uses routine (thinking off). You can "
    "escalate effort after reading evidence; no effort level restricts tools or evidence. Use "
    "reasoning=scientific for all scientific comparisons; routine is for genuine checks or "
    "established input preparation only. Stage is "
    "context, never an action selector: all allowed tools remain available. Then request "
    "the evidence topics needed together as inspect_evidence. List the exact allowed tool "
    "names you intend to compare/prepare in analysis_plan.comparison_tools, including all "
    "scientific alternatives you need to evaluate. For a specific operation, also declare exact "
    "analysis_plan.comparison_operations from evidence_reader.operation_scopes: this distinguishes "
    "reading an existing batch from choosing/preparing NEW computation under the same tool. "
    "The operation scopes are evidence requirements, not a restriction on final routes. "
    "This read is NOT an action proposal. "
    "A source-bound scientific_overview is already supplied here; do not request another "
    "unchanged overview merely to discover the same counts/ranges. Use it to locate the actual "
    "unresolved scientific question, not to prove a candidate-level conclusion. Prefer a direct "
    "full read of the evidence needed now when that scope is clear. Set full_for to the exact "
    "allowed tool names whose complete prerequisites you need NOW (possibly [] for precise "
    "field reads). Merely considering a route does not require preloading all its data. "
    "Explicit view without full_for reads requested topics/fields only. All other tools/evidence "
    "remain available and actual proposals still need their complete candidate pools and critical "
    "prerequisites. In current-pool MC/DFT selection, exact decision policy, versions, limits and "
    "limitations stay loaded, while large historical coverage/task-by-branch components may remain "
    "source-bound previews. Read scientific_decision_brief in full whenever these details can "
    "affect selection, diversity, coverage or confidence. New-region generation/convergence still "
    "requires the original complete coverage evidence. No preview proves scientific sufficiency. "
    "Choose view=overview when the question is still which scientific bottleneck/route needs "
    "attention: overview gives complete counts/availability/ranges, NOT complete candidate "
    "evidence. Do not load every route's full pool just because you considered that route. "
    "Choose view=full directly when the current candidate-level decision is known, requesting "
    "needed fields/topics together; overview is optional, never a required extra call. "
    "Original complete prerequisites are mandatory before accepting the actual scientific "
    "proposal. A provisional choice from overview is checked again after the chosen route's "
    "full evidence is loaded. Choose relevant comparison "
    "routes rather than every tool by default. You may later consider another legal route "
    "and request its evidence; the list does not bind your final choice. Optional fields selects exact "
    "source names from evidence_reader.field_index within those topics. Use flat indexed names, e.g. fields=[qbc_candidates], never dft.qbc_candidates. Do not invent names absent from the index; request the topic if unsure. A dotted subfield request "
    "currently expands to its complete source root (reported in expanded_field_requests), never "
    "a sampled or incomplete candidate pool. Omit fields to load "
    "their default full evidence. Counts and serialized_bytes help size reads; they are not "
    "scientific sufficiency or token estimates. Execution snapshots marked on_demand are "
    "indexed rather than sent automatically; request their exact field whenever relevant. "
    "Fields marked supporting_memory contain prior public explanations or packaging traces, "
    "not a replacement for current observations. Inspect their exact fields when they can affect "
    "the present comparison, causal attribution, costs or return verification; unloaded history "
    "is unknown, not evidence that no prior result or limitation exists. "
    "No evidence is absent merely because it is unloaded. Read full candidate pools before "
    "selection. Request other stages' evidence or escalate reasoning at any time. Reuse "
    "version-matching findings when adequate; reopen them when facts, assumptions or the "
    "question changed. Do not make a scientific choice or execute anything in this read."
)


ANALYSIS_GUIDANCE = (
    "Reuse version-matching observations rather than re-deriving unchanged history. Resolve the "
    "current factual gap before reopening broader strategy, unless scientific uncertainty requires it. "
    "Analyze only the unresolved question in analysis_plan, retaining all confirmed human "
    "constraints and relevant scientific alternatives. It is a revisable working scope, not "
    "a tool restriction or scientific conclusion. If another route or missing evidence matters, "
    "request inspect_evidence with exact fields/topics and revise analysis_plan; reasoning may "
    "be escalated to scientific without rereading unchanged evidence. Unloaded evidence stays "
    "unknown. Never substitute counts or a prior-stage label for candidate-level evidence. "
    "field_overviews are incomplete source-bound previews, not sampled candidate pools or "
    "proof of scientific sufficiency. If provisional_proposal is supplied, reuse its supported "
    "public conclusions and evaluate the newly loaded complete evidence; revise any part when "
    "warranted. It is unvalidated, never an instruction, approval or verified observation."
)

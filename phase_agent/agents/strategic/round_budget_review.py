"""Mandatory two-path trade-off when round results exist, never a scoring rule."""

import math

TOOLS = {"select_dft_candidates": "supplement_dft", "generate_branches": "new_search"}


def round_budget_review_errors(action, context):
    parameters = action.get("parameters") or {}
    if (
        action.get("operation") == "prepare_mlip_relax"
        and (action.get("tool") or action.get("action_type")) == "prepare_local_batch_files"
        and parameters.get("mode") == "relax_inputs"
        and parameters.get("selection_scope") == "candidate_ids"
    ):
        # This prepares a frozen evidence scope; route selection remains downstream.
        return []
    if (action.get("tool") or action.get("action_type")) == "prepare_local_batch_files" and (
        action.get("parameters") or {}
    ).get("mode") in {"model_refresh_inputs", "mc_inputs", "mc_energy_repair_inputs"}:
        # Prepare existing work before deciding the next scientific route.
        return []
    # DIRECT only prepares evidence for later review; it does not choose between
    # DFT and new search or authorize either downstream route.
    if action.get("operation") in {
        "sample_direct_extrapolation",
        "prepare_perturbation_recovery",
        "prepare_parent_first_recovery",
    }:
        return []
    recovery = context.get("extrapolation_recovery") or {}
    decisions = (action.get("parameters") or {}).get("decisions") or []
    selected = [
        row.get("candidate_id")
        for row in decisions
        if isinstance(row, dict) and row.get("action") in {"DFT_SINGLE_POINT", "DFT_RELAX"}
    ]
    expected = recovery.get("selected_candidate_ids") or []
    if (
        action.get("tool") == "select_dft_candidates"
        and recovery.get("status") == "awaiting_reviewed_dft_labels"
        and expected
        and len(selected) == len(expected)
        and set(selected) == set(expected)
    ):
        return []
    if (
        action.get("operation") == "prepare_mlip_training"
        and recovery.get("status") == "awaiting_reviewed_dft_labels"
        and (recovery.get("labels") or {}).get("status") == "complete"
    ):
        return []
    program = action.get("_simulation_program") or {}
    team = action.get("_decision_team") or {}
    approved_input_preparation = (
        (action.get("tool") or action.get("action_type")) == "prepare_local_batch_files"
        and action.get("operation") == "prepare_dft_inputs"
        and parameters == {"mode": "dft_inputs"}
        and bool(action.get("target_ids"))
        and program.get("protocol") == "registered_dft_inputs/v1"
        and program.get("status") == "passed"
        and program.get("target_ids") == action.get("target_ids")
        and team.get("backend") == "codex_native"
        and any(
            event.get("kind") == "human_accept" and event.get("revision") == team.get("revision")
            for event in team.get("human_review") or []
            if isinstance(event, dict)
        )
    )
    if approved_input_preparation:
        # This only prepares the already-selected and reserved tasks; it does
        # not reopen the preceding scientific route or expand its budget.
        return []
    if not (context.get("round_budget_evidence") or {}).get("required"):
        return []
    review = action.get("round_budget_review")
    if not isinstance(review, dict):
        return [
            "round_budget_review: compare supplement_dft and new_search before selecting an action"
        ]
    errors = []
    if review.get("choice") not in {
        "supplement_dft",
        "new_search",
        "prepare_evidence",
        "finetune",
        "stop",
    }:
        errors.append("round_budget_review.choice: unsupported path")
    expected = TOOLS.get(action.get("tool") or action.get("action_type"))
    if expected and review.get("choice") != expected:
        errors.append("round_budget_review.choice: inconsistent with selected tool")
    for key in ("reason", "uncertainty", "revisit_when"):
        if (
            not isinstance(review.get(key), str)
            or not review[key].strip()
            or len(review[key]) > 400
        ):
            errors.append(f"round_budget_review.{key}: concise nonempty text required")
    alternatives = review.get("alternatives")
    if not isinstance(alternatives, list) or len(alternatives) != 2:
        return errors + ["round_budget_review.alternatives: exactly two path assessments required"]
    if {row.get("path") for row in alternatives if isinstance(row, dict)} != {
        "supplement_dft",
        "new_search",
    }:
        errors.append("round_budget_review.alternatives: both paths required")
    for row in alternatives:
        if not isinstance(row, dict):
            errors.append("round_budget_review.alternatives: object required")
            continue
        for key in ("expected_benefit", "cost_basis"):
            if not isinstance(row.get(key), str) or not row[key].strip() or len(row[key]) > 400:
                errors.append(f"round_budget_review.alternatives.{key}: text required")
        cost = row.get("expected_relative_cost")
        if cost is not None and (
            type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0
        ):
            errors.append(
                "round_budget_review.expected_relative_cost: nonnegative finite number or null"
            )
        if row.get("benefit_status") != "hypothesis":
            errors.append(
                "round_budget_review.benefit_status: expected benefits must be hypotheses"
            )
    return errors


def round_budget_instruction():
    return (
        "When round_budget_evidence.required=true include top-level round_budget_review: "
        "choice=supplement_dft/new_search/prepare_evidence/finetune/stop; reason, uncertainty, revisit_when "
        "are concise text. alternatives MUST contain exactly two objects, path=supplement_dft and "
        "path=new_search; each needs expected_benefit, benefit_status=hypothesis, expected_relative_cost "
        "(nonnegative relative_cost estimate or null), cost_basis. Compare marginal total downstream "
        "costs and benefits before candidate selection; cite existing evidence_refs. Missing numerical "
        "cost means null and explicit explanation, never zero. Evidence preparation is allowed when "
        "model refresh/QBC/independent verification is missing. No fixed DFT-first policy. Coverage and error are evidence for expected benefit and uncertainty, NEVER hard gates for new_search. New search may win even with large error or incomplete coverage. Compare complete paths through downstream DFT and any retraining; no fixed accuracy threshold or DFT-first rule. Do not "
        "reprepare completed training solely because job status is stale; report completion is not "
        "model-file verification or activation. Predict benefits, never assert unexecuted outcomes."
    )

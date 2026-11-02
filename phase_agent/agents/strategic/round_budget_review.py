"""Mandatory two-path trade-off when round results exist, never a scoring rule."""

import math

TOOLS = {"select_dft_candidates": "supplement_dft", "generate_branches": "new_search"}


def round_budget_review_errors(action, context):
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

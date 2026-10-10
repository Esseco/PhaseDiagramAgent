"""Low-cost routine transport with lazy, bounded complex proposal analysis."""

from time import perf_counter

ROUTINE_MODES = {"configuration_dialogue", "configuration_json_review"}
PROPOSAL_MODES = {"autonomous_search", "repair_post_dft_action", "round_budget_review_repair"}


def choose_harness(payload):
    if payload.get("mode") in ROUTINE_MODES or payload.get("mode") not in PROPOSAL_MODES:
        return "legacy", "routine_or_nonproposal"
    if payload.get("analysis_harness") == "deepagents":
        return "deepagents", "explicit_complex_analysis"
    context = payload.get("decision_context") or {}
    reports = (
        context.get("training_result_reports")
        or (context.get("round_budget_evidence") or {}).get("training_reports")
        or []
    )
    if context.get("round_budget_evidence") and len(reports) >= 2:
        return "deepagents", "multi_report_budget_review"
    return "legacy", "routine_proposal"


def create_hybrid_client(legacy, deep_factory):
    deep = None

    def call(payload):
        nonlocal deep
        harness, reason = choose_harness(payload)
        start = perf_counter()
        if harness == "deepagents":
            if deep is None:
                deep = deep_factory()
            # Do not silently double-spend with another provider after failure.
            result = deep(payload)
        else:
            result = legacy(payload)
        if not isinstance(result, dict):
            raise TypeError("Invalid proposal response")
        result = dict(result)
        result["_proposal_harness"] = harness
        result["_analysis_route"] = {
            "harness": harness,
            "reason": reason,
            "elapsed_seconds": round(perf_counter() - start, 4),
        }
        return result

    call.decision_backend = "langgraph"
    return call

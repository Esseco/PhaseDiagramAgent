"""Preserve the existing hull-aware QBC selector as a baseline."""

from decision_layer.qbc_selection.select_dft_candidates import select_dft_candidates


def create_qbc_fixed_decisions(candidates: list[dict], *, config: dict, remaining_dft_budget: float) -> dict:
    settings = config.get("fixed") or {}
    result = select_dft_candidates(candidates, batch_size=int(settings.get("batch_size", 10)), seed=int(settings.get("seed", 0)), weights=settings.get("weights"), audit_fraction=float(settings.get("audit_fraction", .1)), cost_budget=remaining_dft_budget)
    selected = {item.get("candidate_id") or item.get("structure_id") for item in result["selected_candidates"]}
    decisions = [{"candidate_id": item.get("candidate_id") or item.get("structure_id"), "action": "DFT_SINGLE_POINT" if (item.get("candidate_id") or item.get("structure_id")) in selected else "DEFER", "reason": "qbc_fixed baseline"} for item in candidates]
    return {"status": "completed", "decisions": decisions, "global_action": "CONTINUE_DATA_COLLECTION", "source": "qbc_fixed", "baseline_result": result}

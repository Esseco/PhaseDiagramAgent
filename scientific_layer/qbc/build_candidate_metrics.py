"""Return objective QBC/context metrics without making DFT decisions."""

from copy import deepcopy


def build_qbc_candidate_metrics(candidates: list[dict], *, qbc_evaluator=None) -> dict:
    rows, failures = [], []
    for candidate in candidates:
        qbc = deepcopy(candidate.get("qbc"))
        if qbc is None and callable(qbc_evaluator):
            qbc = qbc_evaluator(candidate)
        qbc = qbc or {"status": "not_configured"}
        candidate_id = candidate.get("candidate_id") or candidate.get("structure_id")
        if not candidate_id:
            failures.append({"candidate": deepcopy(candidate), "reason": "candidate_id_missing"}); continue
        row = {
            "candidate_id": candidate_id, "branch_id": candidate.get("branch_id"),
            "atom_count": candidate.get("atom_count") if candidate.get("structure") is None else len(candidate["structure"]),
            "f_std_max": qbc.get("f_std_max", qbc.get("force_max_atom_disagreement")),
            "f_std_p95": qbc.get("f_std_p95"), "energy_std": qbc.get("energy_std"),
            "energy_per_atom_std": qbc.get("energy_per_atom_std"),
            "predicted_Ehull": candidate.get("predicted_Ehull", candidate.get("ehull")),
            "composition": deepcopy(candidate.get("composition")), "phase": candidate.get("phase", candidate.get("P")),
            "new_phase_flag": bool(candidate.get("new_phase_flag", False)),
            "distance_to_training_set": candidate.get("distance_to_training_set"),
            "diversity_score": candidate.get("diversity_score"), "hull_impact": candidate.get("hull_impact"),
            "duplicate_of": candidate.get("duplicate_of"), "qbc_status": qbc.get("status"),
            "qbc_interpretation": qbc.get("interpretation", "committee_disagreement_not_true_error"),
            "estimated_cost": deepcopy(candidate.get("estimated_cost")),
        }
        rows.append(row)
    return {"status": "completed" if rows else "insufficient_data", "metrics": rows, "failures": failures, "decision_made": False}

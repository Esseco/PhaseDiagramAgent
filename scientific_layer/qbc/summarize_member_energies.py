"""Model-agnostic QBC statistics from per-model energies."""
import math
import statistics


def summarize_member_energies(energies_per_atom, *, expected_members: int = 3) -> dict:
    values = [float(value) for value in energies_per_atom]
    if len(values) != expected_members or not all(math.isfinite(value) for value in values):
        return {"status": "insufficient_committee", "member_count": len(values),
                "expected_members": expected_members,
                "interpretation": "committee_disagreement_not_true_error"}
    mean = statistics.fmean(values)
    variance = statistics.fmean((value - mean) ** 2 for value in values)
    return {"status": "completed", "member_count": len(values),
            "energy_per_model_per_atom": values, "energy_mean_per_atom": mean,
            "energy_per_atom_std": math.sqrt(variance), "energy_per_atom_var": variance,
            "energy_per_atom_range": max(values) - min(values),
            "interpretation": "committee_disagreement_not_true_error"}

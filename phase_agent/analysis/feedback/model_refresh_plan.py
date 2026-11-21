"""Deterministic, model-isolated refresh selection; no calculation or file writes."""

import math


def build_model_refresh_plan(candidates, *, old_version, new_version, seed=2026, include_all=False):
    """Candidates carry registered IDs, content hashes and old-version evidence.

    Missing energies are evaluated, not silently discarded. Historical structures
    without current energy remain unassessed; all known far structures are predicted.
    """
    if not old_version or not new_version or old_version == new_version:
        raise ValueError("refresh requires two distinct explicit model versions")
    unique = {}
    for candidate in candidates:
        key = candidate.get("structure_sha256")
        if not key or not candidate.get("structure_id"):
            raise ValueError("refresh candidate requires structure ID and content hash")
        row = dict(candidate)
        energy = row.get("ehull")
        if energy is not None:
            if row.get("source_version") != old_version:
                raise ValueError("current ranking evidence must belong to old model")
            if isinstance(energy, bool) or not math.isfinite(float(energy)) or float(energy) < 0:
                raise ValueError("invalid old-model Ehull eV/atom")
            energy = float(energy)
        row["operation"] = "relax" if energy is not None and energy < 0.001 else "predict"
        row["selection_reason"] = (
            "ground_state"
            if row["operation"] == "relax"
            else "unassessed"
            if energy is None
            else "near_hull"
            if energy <= 0.010
            else "far_audit"
        )
        if row.get("hd_required") is True:
            row["selection_reason"] = "high_hd"
        previous = unique.get(key)
        priority = {
            "ground_state": 0,
            "unassessed": 1,
            "near_hull": 2,
            "far_audit": 4,
            "high_hd": -1,
        }
        if (
            previous is None
            or priority[row["selection_reason"]] < priority[previous["selection_reason"]]
        ):
            unique[key] = row
    selected = list(unique.values())
    gaps = []
    selected.sort(key=lambda r: r["structure_sha256"])
    return {
        "old_model_version": old_version,
        "new_model_version": new_version,
        "policy": {
            "relax_below": 0.001,
            "predict_through": 0.010,
            "audit_fraction": 1.0,
            "force_threshold": 0.1,
            "far_force_threshold": 0.2,
            "force_followup_version": 2,
            "maximum_supplemental_waves": 1,
            "seed": seed,
        },
        "candidates": selected,
        "total_unique": len(unique),
        "deferred": len(unique) - len(selected),
        "coverage_gaps": gaps,
        "maximum_supplemental_count": 0,
    }


def supplemental_relaxation_required(ehull, forces):
    """Force criterion uses max atomic vector norm, not component or RMS force."""
    if not math.isfinite(ehull) or ehull < 0:
        raise ValueError("invalid new-model Ehull")
    if ehull < 0.001:
        return True
    if not forces or any(
        len(vector) != 3 or not all(math.isfinite(x) for x in vector) for vector in forces
    ):
        raise ValueError("new-model atomic force vectors missing or invalid")
    threshold = 0.1 if ehull <= 0.010 else 0.2
    return max(math.sqrt(sum(x * x for x in vector)) for vector in forces) > threshold

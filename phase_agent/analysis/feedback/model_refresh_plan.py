"""Deterministic, model-isolated refresh selection; no calculation or file writes."""

from collections import defaultdict
import hashlib
import math


def build_model_refresh_plan(candidates, *, old_version, new_version, seed=2026):
    """Candidates carry registered IDs, content hashes and old-version evidence.

    Missing energies are evaluated, not silently discarded. Historical near-hull
    evidence can promote a structure, but cannot supply a new-version energy.
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
            else "historical_near_hull"
            if row.get("historical_near_hull")
            else "far_audit"
        )
        previous = unique.get(key)
        priority = {
            "ground_state": 0,
            "unassessed": 1,
            "near_hull": 2,
            "historical_near_hull": 3,
            "far_audit": 4,
        }
        if (
            previous is None
            or priority[row["selection_reason"]] < priority[previous["selection_reason"]]
        ):
            unique[key] = row
    selected, strata = [], defaultdict(list)
    for row in unique.values():
        if row["selection_reason"] != "far_audit":
            selected.append(row)
        else:
            strata[(str(row.get("x_Na_per_O2")), str(row.get("phase")))].append(row)
    far = [r for group in strata.values() for r in group]
    target = math.ceil(len(far) * 0.10)
    # Largest remainder allocation gives exactly 10% rounded up, never one per stratum.
    quotas = {k: len(v) * 0.10 for k, v in strata.items()}
    counts = {k: math.floor(q) for k, q in quotas.items()}
    for key in sorted(strata, key=lambda k: (-(quotas[k] - counts[k]), k))[
        : target - sum(counts.values())
    ]:
        counts[key] += 1
    gaps = []
    for key, group in sorted(strata.items()):
        ranked = sorted(
            group,
            key=lambda r: hashlib.sha256(f"{seed}:{r['structure_sha256']}".encode()).hexdigest(),
        )
        selected.extend(ranked[: counts[key]])
        if not counts[key]:
            gaps.append({"na": key[0], "phase": key[1], "deferred": len(group)})
    selected.sort(key=lambda r: r["structure_sha256"])
    return {
        "old_model_version": old_version,
        "new_model_version": new_version,
        "policy": {
            "relax_below": 0.001,
            "predict_through": 0.010,
            "audit_fraction": 0.10,
            "force_threshold": 0.05,
            "maximum_supplemental_waves": 1,
            "seed": seed,
        },
        "candidates": selected,
        "total_unique": len(unique),
        "deferred": len(unique) - len(selected),
        "coverage_gaps": gaps,
        "maximum_supplemental_count": sum(r["operation"] == "predict" for r in selected),
    }


def supplemental_relaxation_required(ehull, forces):
    """Force criterion uses max atomic vector norm, not component or RMS force."""
    if not math.isfinite(ehull) or ehull < 0:
        raise ValueError("invalid new-model Ehull")
    if ehull < 0.001:
        return True
    if ehull > 0.010:
        return False
    if not forces or any(
        len(vector) != 3 or not all(math.isfinite(x) for x in vector) for vector in forces
    ):
        raise ValueError("new-model atomic force vectors missing or invalid")
    return max(math.sqrt(sum(x * x for x in vector)) for vector in forces) > 0.05

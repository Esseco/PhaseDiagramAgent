"""Low-energy prefixes within one phase/composition; flags never establish an energy error."""

from collections import defaultdict
from hashlib import sha256
from statistics import median
from numpy import interp

from phase_agent.analysis.phase.phase_hull_sensitivity import HullSensitivity


def energy_islands(points, groups, limits):
    pools = defaultdict(list)
    for gid, members in groups.items():
        pools[(members[0]["composition_key"], members[0]["phase"])].append((gid, members))
    evaluator = HullSensitivity(points)
    result = []
    for pool in pools.values():
        pool.sort(key=lambda g: min(p["energy_mev_per_O2"] for p in g[1]))
        prefix, gids = [], []
        for index, (gid, members) in enumerate(pool[:-1]):
            prefix.extend(members)
            gids.append(gid)
            if len(gids) > limits["joint_max_order"]:
                break
            upper = max(p["energy_mev_per_O2"] for p in prefix)
            peers = [p for _, group in pool[index + 1 :] for p in group]
            lower = min(p["energy_mev_per_O2"] for p in peers)
            gap = lower - upper
            if gap <= limits["outlier_gap_mev_per_O2"]:
                continue
            reference = [
                p
                for p in peers
                if p["energy_mev_per_O2"] <= lower + limits["reference_window_mev_per_O2"] + 1e-8
            ]
            keys = sorted(p["key"] for p in prefix)
            rows = [r for p in prefix for r in p["rows"]]
            result.append(
                {
                    "cluster_id": "C-" + sha256("|".join(keys).encode()).hexdigest()[:16],
                    "phase": prefix[0]["phase"],
                    "x": prefix[0]["x"],
                    "geometry_keys": keys,
                    "group_ids": list(gids),
                    "group_count": len(gids),
                    "record_ids": sorted({r["record_id"] for r in rows if r.get("record_id")}),
                    "structure_paths": sorted(
                        {r["structure_path"] for r in rows if r.get("structure_path")}
                    ),
                    "reference_record_ids": sorted(
                        {r["record_id"] for p in reference for r in p["rows"] if r.get("record_id")}
                    ),
                    "reference_structure_paths": sorted(
                        {
                            r["structure_path"]
                            for p in reference
                            for r in p["rows"]
                            if r.get("structure_path")
                        }
                    ),
                    "reference_count": len(reference),
                    "reference_status": "available"
                    if len(reference) >= 5
                    else "insufficient_references",
                    "reference_span_mev_per_O2": max(p["energy_mev_per_O2"] for p in reference)
                    - lower,
                    "gap_to_lowest_reference_mev_per_O2": gap,
                    "gap_to_reference_median_mev_per_O2": median(
                        p["energy_mev_per_O2"] for p in reference
                    )
                    - upper,
                    "hull_lift_mev_per_O2": evaluator.lift(keys),
                    "distance_to_hull_mev_per_O2": min(
                        p["energy_mev_per_O2"]
                        - float(interp(p["x"], evaluator.xs, evaluator.baseline))
                        for p in prefix
                    ),
                    "max_distance_to_hull_mev_per_O2": max(
                        p["energy_mev_per_O2"]
                        - float(interp(p["x"], evaluator.xs, evaluator.baseline))
                        for p in prefix
                    ),
                    "energy_error_confirmed": False,
                    "scope": "same_phase_composition_low_energy_prefix_candidate_only",
                }
            )
    # One review item per phase/composition, preferring a supported energy gap.
    selected = {}

    def priority(cluster):
        return (
            cluster["max_distance_to_hull_mev_per_O2"] <= limits["hull_shift_mev_per_O2"],
            cluster["reference_status"] == "available",
            cluster["gap_to_lowest_reference_mev_per_O2"],
        )

    for cluster in result:
        key = (cluster["phase"], cluster["x"])
        rank = priority(cluster)
        previous = selected.get(key)
        if previous is None or rank > priority(previous):
            selected[key] = cluster
    return sorted(
        selected.values(), key=lambda c: (-(c["hull_lift_mev_per_O2"] or 0), c["cluster_id"])
    )

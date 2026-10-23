"""Fixed-snapshot low-energy prefixes; supporting evidence only, no exclusion policy."""

from collections import defaultdict
from hashlib import sha256

import numpy as np

RULE = "static-connected10-prefix5-single-reference-cluster5-floor10-v2"


def evaluate_low_energy_prefixes(points):
    pools = defaultdict(list)
    for point in points:
        pools[(point["composition_key"], point["phase"])].append(point)
    clusters, candidates, by_key = [], [], {}
    for pool in pools.values():
        ordered = sorted(pool, key=lambda p: (p["energy_mev_per_O2"], p["key"]))
        energies = np.array([p["energy_mev_per_O2"] for p in ordered])
        slices = np.split(
            np.arange(len(ordered)), np.flatnonzero(np.diff(energies) > 10 + 1e-8) + 1
        )
        members_by_id, local = {}, []
        for rank, indices in enumerate(slices):
            members = [ordered[i] for i in indices]
            keys = [p["key"] for p in members]
            cid = "DC-" + sha256("|".join(sorted(keys)).encode()).hexdigest()[:16]
            row = {
                "cluster_id": cid,
                "phase": members[0]["phase"],
                "x_Na_per_O2": members[0]["x"],
                "composition_per_O2": dict(members[0]["composition_key"]),
                "cluster_rank": rank,
                "cluster_size": len(members),
                "pool_count": len(pool),
                "energy_min_meV_per_O2": float(energies[indices[0]]),
                "energy_max_meV_per_O2": float(energies[indices[-1]]),
                "geometry_keys": keys,
                "D": None,
                "D_candidate_id": None,
                "D_status": "large_cluster_not_scored"
                if len(members) > 5
                else "outside_low_prefix_budget",
            }
            local.append(row)
            members_by_id[cid] = members
            for key in keys:
                by_key[key] = row
        prefix = []
        for index, current in enumerate(local):
            prefix.extend(members_by_id[current["cluster_id"]])
            if len(prefix) > 5:
                break
            current["D_status"] = "insufficient_single_cluster_reference"
            if index + 1 == len(local):
                continue
            reference_cluster = local[index + 1]
            reference = members_by_id[reference_cluster["cluster_id"]][:5]
            if len(reference) < 2:
                continue
            ref_energies = [p["energy_mev_per_O2"] for p in reference]
            typical = float(np.median(np.diff(ref_energies)))
            gap = ref_energies[0] - prefix[-1]["energy_mev_per_O2"]
            keys = [p["key"] for p in prefix]
            identity = "DP-" + sha256("|".join(sorted(keys)).encode()).hexdigest()[:16]
            candidate = {
                "candidate_id": identity,
                "phase": current["phase"],
                "x_Na_per_O2": current["x_Na_per_O2"],
                "geometry_keys": keys,
                "cluster_ids": [c["cluster_id"] for c in local[: index + 1]],
                "candidate_size": len(prefix),
                "reference_cluster_id": reference_cluster["cluster_id"],
                "reference_geometry_keys": [p["key"] for p in reference],
                "reference_count": len(reference),
                "reference_few": len(reference) < 5,
                "reference_energies_meV_per_O2": ref_energies,
                "reference_spacings_meV_per_O2": np.diff(ref_energies).tolist(),
                "reference_typical_spacing_meV_per_O2": typical,
                "gap_meV_per_O2": gap,
                "denominator_meV_per_O2": max(10.0, typical),
                "D": gap / max(10.0, typical),
            }
            candidates.append(candidate)
            for cluster in local[: index + 1]:
                if cluster["D"] is None or candidate["D"] > cluster["D"] + 1e-12:
                    cluster.update(D=candidate["D"], D_candidate_id=identity, D_status="available")
        clusters.extend(local)
    return clusters, candidates, by_key

"""Keep an endpoint phase baseline unless an independent same-phase band remains."""

from collections import defaultdict


def endpoint_guards(points, policy, removed=()):
    """Return guarded baseline keys, including guards induced by a proposed joint removal.

    Tied/near-tied members of the baseline are not independent replacement evidence.
    Reuse the existing reference requirement: five exact geometries within 50 meV/O2.
    This is an evidence sufficiency rule, not a claim that the reference is correct.
    """
    if not points:
        return {}
    ends = {min(p["x"] for p in points), max(p["x"] for p in points)}
    pools = defaultdict(list)
    for point in points:
        if point["x"] in ends:
            pools[(point["composition_key"], point["phase"])].append(point)
    removed = set(removed)
    guards = {}
    for pool in pools.values():
        pool = sorted(pool, key=lambda p: (p["energy_mev_per_O2"], p["key"]))
        minimum = pool[0]["energy_mev_per_O2"]
        baseline = [
            p
            for p in pool
            if p["energy_mev_per_O2"] <= minimum + policy["group_span_mev_per_O2"] + 1e-8
        ]
        baseline_keys = {p["key"] for p in baseline}
        peers = [p for p in pool if p["key"] not in baseline_keys | removed]
        window = policy["reference_window_mev_per_O2"]
        supported = any(
            peers[i + 4]["energy_mev_per_O2"] - peers[i]["energy_mev_per_O2"] <= window + 1e-8
            for i in range(len(peers) - 4)
        )
        if not supported:
            for point in baseline:
                guards[point["key"]] = (
                    "端点同相独立替代参考不足；保留基准簇，待人工验证或补算，其他相不能替代"
                )
    return guards


def assert_endpoint_removal(points, policy, removed):
    if set(removed).intersection(endpoint_guards(points, policy, removed)):
        raise ValueError("不能移除端点同相基准簇：缺少保留下来的独立同相替代参考")


def retained_keys(state, snapshot, policy, points):
    """Bind an explicit retain to this snapshot, policy and exact geometry."""
    decisions = state.get("phase_quality_decisions") or []
    return {
        p["key"]
        for p in points
        if any(
            d.get("decision") == "retain"
            and d.get("model_version") == snapshot.get("model_version")
            and d.get("hull_version") == snapshot.get("version")
            and d.get("screening_policy") == policy
            and d.get("geometry_key") == p["key"]
            and sorted(d.get("record_ids", [])) == sorted(r["record_id"] for r in p["rows"])
            for d in decisions
        )
    }


def select_preview_keys(points, policy, cases, islands, *, keep=()):
    """Take whole eligible sets along one deterministic, endpoint-safe exclusion chain."""
    targets = [
        c for c in cases if (c.get("hull_lift_mev_per_O2") or 0) > policy["hull_shift_mev_per_O2"]
    ] + [
        c
        for c in islands
        if c["distance_to_hull_mev_per_O2"] <= policy["hull_shift_mev_per_O2"]
        and c["gap_to_lowest_reference_mev_per_O2"] > policy["outlier_gap_mev_per_O2"]
    ]
    targets.sort(
        key=lambda c: (
            -(c.get("gap_to_lowest_reference_mev_per_O2") or 0),
            -(c.get("hull_lift_mev_per_O2") or 0),
            tuple(c["geometry_keys"]),
        )
    )
    removed, blocked = set(), set()
    from phase_agent.analysis.phase.phase_hull_sensitivity import HullSensitivity
    from numpy import interp

    hull = HullSensitivity(points)
    near = {
        p["key"]
        for p in points
        if p["energy_mev_per_O2"] - float(interp(p["x"], hull.xs, hull.baseline))
        <= policy["hull_shift_mev_per_O2"] + 1e-8
    }
    for target in targets:
        keys = set(target["geometry_keys"])
        if not keys <= near:
            continue  # Reconsider the whole set only after its far members become near-hull.
        proposed = removed | keys
        guards = set(endpoint_guards(points, policy, proposed))
        if keys.intersection(keep) or proposed.intersection(guards):
            blocked.update(keys.intersection(guards))
            continue
        removed = proposed
    return removed, blocked

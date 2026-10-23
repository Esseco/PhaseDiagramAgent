"""Exact small-removal hull queries on a cached composition grid."""

from itertools import combinations
import numpy as np


class HullSensitivity:
    """Each unit is one geometry or one same-composition energy group."""

    def __init__(self, points, unit_by_key=None):
        self.points = points
        self.unit_by_key = unit_by_key or {p["key"]: p["key"] for p in points}
        self.xs = np.array(sorted({p["x"] for p in points}), dtype=float)
        x_index = {x: i for i, x in enumerate(self.xs)}
        pools = [{} for _ in self.xs]
        self.positions = {}
        for p in points:
            index = x_index[p["x"]]
            unit = self.unit_by_key[p["key"]]
            pools[index][unit] = min(p["energy_mev_per_O2"], pools[index].get(unit, float("inf")))
            self.positions.setdefault(unit, set()).add(index)
        self.pools = [sorted((energy, unit) for unit, energy in pool.items()) for pool in pools]
        self.ys = np.array([pool[0][0] for pool in self.pools])
        self.baseline = self._envelope(self.xs, self.ys)
        self.cache = {}
        self.envelope_evaluations = 0

    @staticmethod
    def _vertices(xs, ys):
        hull = []
        for x, y in zip(xs, ys):
            while len(hull) >= 2:
                a, b = hull[-2:]
                if (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]) > 1e-14:
                    break
                hull.pop()
            hull.append((x, y))
        return hull

    @classmethod
    def _envelope(cls, xs, ys):
        hull = cls._vertices(xs, ys)
        return np.interp(xs, [p[0] for p in hull], [p[1] for p in hull])

    def lift(self, removed):
        removed = set(removed)
        positions = (
            set().union(*(self.positions.get(u, set()) for u in removed)) if removed else set()
        )
        changes = []
        for index in sorted(positions):
            replacement = next(
                (energy for energy, unit in self.pools[index] if unit not in removed), float("inf")
            )
            if replacement != self.ys[index]:
                changes.append((index, replacement))
        signature = tuple(changes)
        if signature in self.cache:
            return self.cache[signature]
        if not changes:
            return 0.0
        values = self.ys.copy()
        for index, energy in changes:
            values[index] = energy
        if not np.isfinite(values[0]) or not np.isfinite(values[-1]):
            self.cache[signature] = None
            return None
        mask = np.isfinite(values)
        ys = self._envelope(self.xs[mask], values[mask])
        alternate = np.interp(self.xs, self.xs[mask], ys)
        result = max(0.0, float(np.max(alternate - self.baseline)))
        self.cache[signature] = result
        self.envelope_evaluations += 1
        return result

    def next_units(self, removed, remaining_order):
        """Only neighboring support can create a new minimal joint effect in 1D."""
        removed = set(removed)
        pools = [
            [(energy, unit) for energy, unit in pool if unit not in removed] for pool in self.pools
        ]
        mask = np.array([bool(pool) for pool in pools])
        xs = self.xs[mask]
        ys = np.array([pool[0][0] for pool in pools if pool])
        vertices = self._vertices(xs, ys)
        if removed:
            positions = set().union(*(self.positions[u] for u in removed))
            left, right = min(self.xs[i] for i in positions), max(self.xs[i] for i in positions)
            before = [p for p in vertices if p[0] < left]
            after = [p for p in vertices if p[0] > right]
            vertices = [p for p in vertices if left <= p[0] <= right]
            vertices += before[-1:] + after[:1]
        index_by_x = {x: i for i, x in enumerate(self.xs)}
        result = set()
        for x, energy in vertices:
            tied = [u for e, u in pools[index_by_x[x]] if abs(e - energy) <= 1e-5]
            # More tied supports than the removal budget cannot be a minimal cause.
            if len(tied) <= remaining_order:
                result.update(tied)
        return sorted(result)


def joint_cases(
    points,
    groups,
    *,
    threshold,
    order=3,
    keep_per_order=None,
    escalation_threshold=None,
    strict_review=False,
):
    """Search review and strict escalation independently, sharing the grid-query cache."""
    targets = [(threshold, strict_review)]
    if escalation_threshold is not None:
        targets.append((escalation_threshold, True))
    evaluators = {}
    merged, stats, recursive_groups = {}, {}, []
    for cutoff, strict in targets:
        result = _minimal_joint_cases(
            points,
            groups,
            threshold=cutoff,
            order=order,
            keep_per_order=keep_per_order,
            strict=strict,
            evaluators=evaluators,
        )
        if cutoff == threshold:
            recursive_groups = result["recursive_group_ids"]
        for mode, counts in result["stats"].items():
            stats[f"{mode}:{'>' if strict else '>='}{cutoff:g}"] = counts
        for case in result["cases"]:
            identity = (case["mode"], tuple(case["unit_ids"]))
            if identity not in merged:
                merged[identity] = {**case, "minimal_for_thresholds_mev_per_O2": []}
            merged[identity]["minimal_for_thresholds_mev_per_O2"].append(cutoff)
    # Preserve the historical low-level stats interface when called for one threshold.
    if escalation_threshold is None:
        stats = result["stats"]
    return {
        "cases": sorted(
            merged.values(), key=lambda c: (-c["hull_lift_mev_per_O2"], c["mode"], c["unit_ids"])
        ),
        "stats": stats,
        "recursive_group_ids": recursive_groups,
        "max_order": order,
        "algorithm": "composition_grid_cache_and_local_branch_and_bound",
        "scope": f"independent minimal sets for >= {threshold:g} and strict escalation {escalation_threshold}; up to {order} units",
    }


def _minimal_joint_cases(points, groups, *, threshold, order, keep_per_order, strict, evaluators):
    """Check minimal removal sets up to the requested order, with local pruning."""
    cases, stats, minimal_groups = [], {}, set()
    mapping = {p["key"]: gid for gid, members in groups.items() for p in members}
    members_by_key = {p["key"]: p for p in points}
    for mode, unit_map in (("geometry", None), ("energy_group", mapping)):
        if mode not in evaluators:
            evaluators[mode] = HullSensitivity(points, unit_map)
        evaluator = evaluators[mode]
        evaluations_before = evaluator.envelope_evaluations
        high_subsets, visited = set(), set()
        found = []
        counts = {
            "candidate_units": len(evaluator.positions),
            "queries": 0,
            "high_impact": 0,
            "unknown_endpoint": 0,
            "reported": 0,
        }
        frontier = [()]
        for size in range(1, order + 1):
            following = []
            for prior in frontier:
                for unit in evaluator.next_units(prior, order - len(prior)):
                    combo = tuple(sorted((*prior, unit)))
                    if combo in visited:
                        continue
                    visited.add(combo)
                    if any(
                        frozenset(subset) in high_subsets
                        for n in range(1, size)
                        for subset in combinations(combo, n)
                    ):
                        continue  # A known high-impact subset already explains this superset.
                    counts["queries"] += 1
                    lift = evaluator.lift(combo)
                    if lift is None:
                        counts["unknown_endpoint"] += 1
                        continue
                    high = lift > threshold if strict else lift + 1e-7 >= threshold
                    if not high:
                        following.append(combo)
                        continue
                    counts["high_impact"] += 1
                    high_subsets.add(frozenset(combo))
                    keys = sorted(
                        {
                            key
                            for unit in combo
                            for key in (
                                [unit] if mode == "geometry" else [p["key"] for p in groups[unit]]
                            )
                        }
                    )
                    if mode == "energy_group":
                        minimal_groups.update(combo)
                    found.append(
                        {
                            "mode": mode,
                            "order": size,
                            "unit_ids": list(combo),
                            "geometry_keys": keys,
                            "hull_lift_mev_per_O2": lift,
                            "minimal_high_impact": True,
                            "record_ids": sorted(
                                {
                                    r["record_id"]
                                    for key in keys
                                    for r in members_by_key[key]["rows"]
                                    if r.get("record_id")
                                }
                            ),
                            "structure_paths": sorted(
                                {
                                    r["structure_path"]
                                    for key in keys
                                    for r in members_by_key[key]["rows"]
                                    if r.get("structure_path")
                                }
                            ),
                        }
                    )
            frontier = following
        for size in range(1, order + 1):
            selected = sorted(
                [c for c in found if c["order"] == size],
                key=lambda c: (-c["hull_lift_mev_per_O2"], c["unit_ids"]),
            )
            cases.extend(selected if keep_per_order is None else selected[:keep_per_order])
        counts["envelope_evaluations"] = evaluator.envelope_evaluations - evaluations_before
        counts["reported"] = sum(c["mode"] == mode for c in cases)
        stats[mode] = counts
    return {
        "cases": cases,
        "stats": stats,
        "recursive_group_ids": sorted(minimal_groups),
        "max_order": order,
        "algorithm": "composition_grid_cache_and_local_branch_and_bound",
        "scope": f"minimal high-impact sets up to {order} units; supersets of an already high-impact subset are not repeated",
    }

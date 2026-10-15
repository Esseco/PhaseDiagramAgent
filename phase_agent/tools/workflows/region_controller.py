"""Approved branch generation in bounded rounds with explicit novelty semantics."""

from collections import Counter
from copy import copy, deepcopy
from time import monotonic

from phase_agent.tools.policy.region_sampling import CONTROLS, branch_key, errors


class Diversity:
    """Initial TM nearest-shell correlations; not energies or physical convergence."""

    def __init__(self, references):
        self.references, self.graphs = references, {}

    def __call__(self, row):
        key = branch_key(row)
        geometry = key[:2]
        if geometry not in self.graphs:
            from phase_agent.science.structures.boundary_utils import load_structure
            from pymatgen.core import Element

            structure = load_structure(self.references[row["P"]]).copy()
            structure.make_supercell(row["H"])
            sites = [s for s in structure if Element(s.specie.symbol).is_transition_metal]
            sites.sort(
                key=lambda s: (
                    round(float(s.frac_coords[2] % 1), 8),
                    round(float(s.frac_coords[1] % 1), 8),
                    round(float(s.frac_coords[0] % 1), 8),
                )
            )
            pairs = [
                (
                    i,
                    j,
                    structure.lattice.get_distance_and_image(a.frac_coords, b.frac_coords)[0],
                )
                for i, a in enumerate(sites)
                for j, b in enumerate(sites)
                if j > i
            ]
            distances = [d for _, _, d in pairs if d > 1e-6]
            cutoff = min(distances) * 1.1 if distances else 0
            self.graphs[geometry] = (
                len(sites),
                [(i, j) for i, j, d in pairs if 1e-6 < d <= cutoff],
            )
        count, edges = self.graphs[geometry]
        if len(row["T"]) != count:
            raise ValueError("TM排列长度与母结构超胞不一致")
        histogram = Counter(tuple(sorted((row["T"][i], row["T"][j]))) for i, j in edges)
        descriptor = tuple(
            sorted(
                (pair, round(value / max(1, len(edges)) / 0.1)) for pair, value in histogram.items()
            )
        )
        return key[:3], descriptor


def sample_regions(
    manager,
    references,
    params,
    *,
    system_config=None,
    strategy_options=None,
    producer=None,
    descriptor=None,
    clock=monotonic,
):
    problems = errors(params)
    if problems:
        raise ValueError(";".join(problems))
    if not params["regions"]:
        return [], {
            "status": "completed",
            "stop_reason": "explicit_defer",
            "attempts": 0,
            "accepted_before_final_dedup": 0,
            "regions": [],
        }
    from phase_agent.science.structures.propose_branches import propose_branches

    started = clock()
    framework_options = {}
    if producer is None:
        from phase_agent.science.structures.enumerate_legal_frameworks import (
            enumerate_legal_frameworks,
        )
        from fractions import Fraction
        import json

        allowed = set()
        for region in params.get("region_preview", {}).get("regions", []):
            phase = next(
                r["phase"] for r in params["regions"] if r["region_id"] == region["region_id"]
            )
            allowed.update(
                (
                    phase,
                    json.dumps(c["H"], separators=(",", ":")),
                    str(Fraction(c["x"])),
                )
                for c in region["cells"]
            )
        if not allowed and params["regions"]:
            raise ValueError("受控生成必须使用已冻结的合法区域元数据")
        frameworks = []
        for row in enumerate_legal_frameworks(manager.boundary, references):
            xs = [
                x
                for x in row["allowed_x"]
                if (
                    row["P"],
                    json.dumps(row["H"], separators=(",", ":")),
                    str(Fraction(x)),
                )
                in allowed
            ]
            if xs:
                frameworks.append({**row, "allowed_x": xs})
        framework_options["framework_enumerator"] = lambda *a: frameworks
    producer = producer or propose_branches
    descriptor = descriptor or Diversity(references)
    shadow = copy(manager)
    shadow.data = deepcopy(manager.data)
    parents = sorted(manager.data["branches"])
    known = {branch_key(row) for row in manager.data["branches"].values()}
    diversity = {descriptor(row) for row in manager.data["branches"].values()}
    controls = params.get("sampling_controls") or CONTROLS
    regions = sorted(params["regions"], key=lambda r: (r["priority"], r["region_id"]))
    reports = {
        r["region_id"]: {
            "region_id": r["region_id"],
            "attempts": 0,
            "accepted": 0,
            "duplicates": 0,
            "redundant_descriptor": 0,
            "stagnant_batches": 0,
            "stop_reason": None,
        }
        for r in regions
    }
    chosen, attempts, batches = [], 0, 0
    stop = "regions_saturated_under_descriptor"
    while any(reports[r["region_id"]]["stop_reason"] is None for r in regions):
        for region in regions:
            report = reports[region["region_id"]]
            if report["stop_reason"]:
                continue
            if attempts >= controls["max_attempts"]:
                stop = "paused_attempt_guard"
                break
            if len(chosen) >= controls["review_after_new_branches"]:
                stop = "paused_branch_review_guard"
                break
            if clock() - started >= controls["max_seconds"]:
                stop = "paused_time_guard"
                break
            work = min(controls["batch_attempts"], controls["max_attempts"] - attempts)
            produced = producer(
                shadow,
                references,
                quotas={region["strategy"]: work},
                seed=params.get("seed", 0) + batches * 10000,
                parent_branch_ids=parents,
                register=False,
                system_config=system_config,
                strategy_options={**(strategy_options or {}), "_focus": region},
                max_det_H=params.get("max_det_H"),
                **framework_options,
            )
            batches += 1
            attempts += work
            report["attempts"] += work
            gained = 0
            for item in produced[:work]:
                key = branch_key(item)
                if key in known:
                    report["duplicates"] += 1
                    continue
                known.add(key)
                shadow.add_branch(
                    P=item["P"],
                    H=item["H"],
                    x=item["x"],
                    T=item["T"],
                    composition=item.get("composition"),
                )
                feature = descriptor(item)
                if feature in diversity:
                    report["redundant_descriptor"] += 1
                    continue
                if len(chosen) >= controls["review_after_new_branches"]:
                    stop = "paused_branch_review_guard"
                    break
                diversity.add(feature)
                chosen.append({**item, "sampling_region_id": region["region_id"]})
                report["accepted"] += 1
                gained += 1
            report["stagnant_batches"] = 0 if gained else report["stagnant_batches"] + 1
            if report["stagnant_batches"] >= controls["stagnant_batches"]:
                report["stop_reason"] = "no_new_coverage_or_descriptor_in_consecutive_batches"
            if stop.startswith("paused_"):
                break
        if stop.startswith("paused_"):
            break
    for report in reports.values():
        report["stop_reason"] = report["stop_reason"] or stop
    return chosen, {
        "status": "paused_for_review" if stop.startswith("paused_") else "completed",
        "stop_reason": stop,
        "attempts": attempts,
        "accepted_before_final_dedup": len(chosen),
        "regions": list(reports.values()),
        "limitation": "初始TM近邻相关性是便宜的多样性代理，不代表能量或物理收敛；随机停滞不证明穷尽空间。",
    }

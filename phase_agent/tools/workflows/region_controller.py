"""Approved branch generation in bounded rounds with explicit novelty semantics."""

from collections import Counter
from copy import copy, deepcopy
from time import monotonic

from phase_agent.policies.initial_coverage import cell_key

from phase_agent.policies.region_sampling import CONTROLS, branch_key, errors, sampling_limit


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
    from phase_agent.science.structures.generate_branch_structure import sampling_structure_cache

    with sampling_structure_cache() as cache:
        chosen, report = _sample_regions(
            manager,
            references,
            params,
            system_config=system_config,
            strategy_options=strategy_options,
            producer=producer,
            descriptor=descriptor,
            clock=clock,
        )
    report["fixed_tm_structure_cache"] = {"hits": cache["hits"], "misses": cache["misses"]}
    return chosen, report


def _sample_regions(
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
        # Frozen preview already checked legality; group its metadata without expanding cells.
        from fractions import Fraction
        import json

        grouped = {}
        for region in params.get("region_preview", {}).get("regions", []):
            phase = next(
                r["phase"] for r in params["regions"] if r["region_id"] == region["region_id"]
            )
            for cell in region["cells"]:
                key = (phase, json.dumps(cell["H"]))
                row = grouped.setdefault(
                    key, {"P": phase, "H": cell["H"], "det_H": cell["det_H"], "allowed_x": set()}
                )
                row["allowed_x"].add(str(Fraction(cell["x"])))
        frameworks = [
            {**row, "allowed_x": sorted(row["allowed_x"], key=Fraction)} for row in grouped.values()
        ]
        if not frameworks:
            raise ValueError("受控生成必须使用已冻结的合法区域元数据")
        framework_options["framework_enumerator"] = lambda *a: frameworks
    producer = producer or propose_branches
    if descriptor is None:
        tm_ratio = manager.boundary.get("TM_ratio") or {}
        descriptor = (
            (lambda row: branch_key(row)[:3]) if len(tm_ratio) == 1 else Diversity(references)
        )
    shadow = manager
    parents = sorted(manager.data["branches"])
    from phase_agent.evidence.near_hull_parents import PARENT_STRATEGIES

    parent_snapshot = (params.get("region_preview") or {}).get("near_hull_parent_evidence") or {}
    evidence_by_parent = {row["branch_id"]: row for row in parent_snapshot.get("parents") or []}
    if (
        producer is propose_branches
        and any(r["strategy"] in PARENT_STRATEGIES for r in params["regions"])
        and not evidence_by_parent
    ):
        raise ValueError("near_hull_parent_evidence_required")
    controls = params.get("sampling_controls") or CONTROLS
    max_new_branches = sampling_limit(params, controls)
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
    initial = (params.get("region_preview") or {}).get("initial_coverage") or {}
    expected = (
        {
            cell_key(
                next(r["phase"] for r in params["regions"] if r["region_id"] == row["region_id"]),
                cell["H"],
                cell["x"],
            )
            for row in (params.get("region_preview") or {}).get("regions", [])
            for cell in row["cells"]
        }
        if initial.get("required")
        else set()
    )
    covered = set()
    chosen, attempts, batches = [], 0, 0
    stop = "requested_samples_complete"
    while (
        any(reports[r["region_id"]]["stop_reason"] is None for r in regions)
        and stop != "initial_coverage_complete"
    ):
        shares = params.get("strategy_shares")
        active = [r for r in regions if reports[r["region_id"]]["stop_reason"] is None]
        if shares:
            counts = {
                strategy: sum(
                    reports[r["region_id"]]["accepted"]
                    for r in regions
                    if r["strategy"] == strategy
                )
                for strategy in shares
            }
            active = [
                min(
                    active,
                    key=lambda r: (
                        counts[r["strategy"]] / shares[r["strategy"]],
                        reports[r["region_id"]]["attempts"],
                        r["priority"],
                        r["region_id"],
                    ),
                )
            ]
        for region in active:
            report = reports[region["region_id"]]
            if report["stop_reason"]:
                continue
            if attempts >= controls["max_attempts"]:
                stop = "paused_attempt_guard"
                break
            if attempts >= max_new_branches:
                stop = "sampling_budget_complete"
                break
            if clock() - started >= controls["max_seconds"]:
                stop = "paused_time_guard"
                break
            remaining = region["sample_count"] - report["attempts"]
            if remaining <= 0:
                report["stop_reason"] = "requested_samples_complete"
                continue
            work = min(
                remaining,
                controls["batch_attempts"],
                controls["max_attempts"] - attempts,
                max_new_branches - attempts,
            )
            if work <= 0:
                stop = "sampling_budget_complete"
                break
            if shares:
                # Small reviews must give low-share strategies a chance before the guard fills.
                work = min(work, max(1, int(max_new_branches * shares[region["strategy"]] / 2)))
            produced = producer(
                shadow,
                references,
                quotas={region["strategy"]: work},
                seed=params.get("seed", 0) + batches * 10000,
                parent_branch_ids=(
                    sorted(evidence_by_parent)
                    if region["strategy"] in PARENT_STRATEGIES and evidence_by_parent
                    else parents
                ),
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
                covered.add(key[:3])
                source = evidence_by_parent.get(item.get("parent_branch_id"))
                chosen.append(
                    {
                        **item,
                        "sampling_region_id": region["region_id"],
                        "near_hull_parent_evidence": source,
                    }
                )
                report["accepted"] += 1
                gained += 1
            if report["attempts"] >= region["sample_count"]:
                report["stop_reason"] = "requested_samples_complete"
            if len(produced) < work:
                report["stop_reason"] = "insufficient_candidates"
            report["stagnant_batches"] = 0 if gained else report["stagnant_batches"] + 1
            if report["stagnant_batches"] >= controls["stagnant_batches"]:
                report["stop_reason"] = "no_new_coverage_or_descriptor_in_consecutive_batches"
            if stop.startswith("paused_") or stop in {
                "initial_coverage_complete",
                "sampling_budget_complete",
            }:
                break
        if stop.startswith("paused_") or stop in {
            "initial_coverage_complete",
            "sampling_budget_complete",
        }:
            break
    if not chosen and not stop.startswith("paused_"):
        stop = "paused_no_candidates"
    for report in reports.values():
        report["stop_reason"] = report["stop_reason"] or stop
    requested = {row["region_id"]: row["sample_count"] for row in regions}
    for report in reports.values():
        report["requested"] = requested[report["region_id"]]
        report["unfilled"] = max(0, report["requested"] - report["accepted"])
    unfilled = sum(report["unfilled"] for report in reports.values())
    if (
        stop == "requested_samples_complete"
        and unfilled
        and not (initial.get("required") and expected <= covered)
    ):
        stop = "regions_finished_with_shortfall"
    return chosen, {
        "initial_coverage": {
            **initial,
            "complete": expected <= covered,
            "covered_cells": len(covered),
            "missing_cells": len(expected - covered),
        }
        if initial.get("required")
        else {"required": False},
        "status": "paused_for_review" if stop.startswith("paused_") else "completed",
        "stop_reason": stop,
        "attempts": attempts,
        "accepted_before_final_dedup": len(chosen),
        "requested_candidate_count": sum(requested.values()),
        "unfilled_requested_candidates": unfilled,
        "strategy_shares_target": params.get("strategy_shares"),
        "strategy_accepted_counts": {
            strategy: sum(
                reports[r["region_id"]]["accepted"] for r in regions if r["strategy"] == strategy
            )
            for strategy in sorted({r["strategy"] for r in regions})
        },
        "regions": list(reports.values()),
        "limitation": "候选按预算生成后统一去重；未覆盖区域不代表空间已穷尽。",
    }

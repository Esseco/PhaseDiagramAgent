"""Bounded, read-only micro-evidence drill-down queries for expert agents."""

from __future__ import annotations

from collections import defaultdict
import statistics
from typing import Any

ALLOWED_DRILLDOWN_QUERIES = (
    "phase_error_breakdown",
    "sampling_weight_stats",
    "extrapolation_hotspots",
)


def _phase_error_breakdown(state: dict[str, Any], max_rows: int) -> list[dict[str, Any]]:
    by_phase: dict[str, list[float]] = defaultdict(list)
    records = state.get("dft_dataset_records") or []
    tasks = {t.get("task_id"): t for t in state.get("tasks") or [] if t.get("task_id")}
    for rec in records:
        if rec.get("status") != "completed":
            continue
        pred = rec.get("mlip_prediction") or {}
        if pred.get("status") != "completed":
            continue
        n = int(rec.get("atom_count") or 0)
        e_dft, e_mlip = rec.get("energy"), pred.get("energy")
        if n <= 0 or e_dft is None or e_mlip is None:
            continue
        task = tasks.get(rec.get("task_id")) or {}
        phase = str(
            rec.get("actual_phase") or task.get("actual_phase") or task.get("phase") or "unknown"
        )
        by_phase[phase].append(abs(float(e_mlip) - float(e_dft)) / n)
    rows = [
        {
            "phase": phase,
            "count": len(vals),
            "energy_mae_eV_per_atom": round(statistics.fmean(vals), 6),
        }
        for phase, vals in by_phase.items()
    ]
    rows.sort(key=lambda r: (-r["energy_mae_eV_per_atom"], r["phase"]))
    return rows[:max_rows]


def _sampling_weight_stats(state: dict[str, Any], max_rows: int) -> dict[str, Any]:
    sampling = state.get("remote_dft_sampling") or {}
    if not sampling:
        return {"status": "unavailable"}
    latest = list(sampling.values())[-1]
    selected = latest.get("selected") or []
    weights = [
        float(row["weight"]) for row in selected if isinstance(row.get("weight"), (int, float))
    ]
    by_branch: dict[str, int] = defaultdict(int)
    for row in selected:
        by_branch[str(row.get("branch_id") or "unknown")] += 1
    top_branches = sorted(by_branch.items(), key=lambda x: (-x[1], x[0]))[:max_rows]
    return {
        "status": latest.get("status"),
        "gs_count": latest.get("gs_count", 0),
        "kmeans_count": latest.get("kmeans_count", 0),
        "excluded_hull_outlier_count": latest.get("excluded_hull_outlier_count", 0),
        "weight_min": round(min(weights), 6) if weights else None,
        "weight_median": round(statistics.median(weights), 6) if weights else None,
        "weight_max": round(max(weights), 6) if weights else None,
        "top_branches": [{"branch_id": b, "count": c} for b, c in top_branches],
    }


def _extrapolation_hotspots(state: dict[str, Any], max_rows: int) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for task in state.get("tasks") or []:
        output = task.get("outputs") or {}
        monitor = output.get("extrapolation_monitor") or {}
        failed = monitor.get("extrapolation_failure") is True or bool(
            output.get("extrapolation_failures")
        )
        if not failed:
            continue
        phase = str(
            task.get("actual_phase") or task.get("source_phase") or task.get("phase") or "unknown"
        )
        stage = str(task.get("stage") or "unknown")
        counts[(phase, stage)] += 1
    rows = [
        {"phase": phase, "stage": stage, "failure_count": count}
        for (phase, stage), count in counts.items()
    ]
    rows.sort(key=lambda r: (-r["failure_count"], r["phase"], r["stage"]))
    return rows[:max_rows]


def run_drilldown_queries(
    snapshot: dict[str, Any],
    queries: list[str],
    *,
    max_rows: int = 10,
) -> dict[str, Any]:
    """Execute up to 2 bounded read-only micro-queries against the frozen snapshot."""
    facts = snapshot.get("facts") or {}
    state = facts.get("state") or {}
    results: dict[str, Any] = {}
    for name in (queries or [])[:2]:
        if name == "phase_error_breakdown":
            results[name] = _phase_error_breakdown(state, max_rows)
        elif name == "sampling_weight_stats":
            results[name] = _sampling_weight_stats(state, max_rows)
        elif name == "extrapolation_hotspots":
            results[name] = _extrapolation_hotspots(state, max_rows)
        else:
            results[name] = {"error": "unknown_query"}
    return results

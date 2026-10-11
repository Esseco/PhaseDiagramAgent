"""Same-frame error diagnostics and narrowly triggered competing-phase checks."""

from collections import defaultdict
import json
from math import isclose, isfinite, sqrt

from pymatgen.core import Composition

from phase_agent.analysis.feedback.dft_comparison_tables import (
    build_comparison_tables,
    comparison_metrics,
)


def single_point_validation(state, *, model_version, minimum_checks, phase_gap=None):
    records = [
        row
        for row in state.get("dft_dataset_records", [])
        if row.get("model_version") == model_version and row.get("stage") == "dft_single_point"
    ]
    comparisons = state.get("dft_mlip_comparisons") or []
    energies, forces = build_comparison_tables(records, comparisons, round_name="convergence")
    paired = [row for row in energies if row["comparison_status"] == "completed"]
    # Task identities, not sorted table order, determine the most recent sample.
    by_task = {row["task_id"]: row for row in paired}
    recent_ids = list(dict.fromkeys(row["task_id"] for row in records if row["task_id"] in by_task))
    selected = [by_task[task_id] for task_id in recent_ids]
    selected_forces = [row for row in forces if row["task_id"] in recent_ids]
    metrics = comparison_metrics(selected, selected_forces, {"model_version": model_version})
    vectors = defaultdict(list)
    for row in selected_forces:
        vectors[(row["task_id"], row["atom_index"])].append(row["dft_force_eV_per_A"])
    metrics["dft_max_residual_force_ev_per_angstrom"] = max(
        (sqrt(sum(value * value for value in vector)) for vector in vectors.values()),
        default=None,
    )
    metrics["task_ids"] = recent_ids
    metrics["enough_checks"] = minimum_checks > 0 and len(selected) >= minimum_checks
    competition = (
        competing_phase_validation(state, paired, model_version=model_version, gap=phase_gap)
        if phase_gap is not None
        else {"status": "disabled", "passed": True}
    )
    return {"metrics": metrics, "phase_competition": competition}


def competing_phase_validation(state, paired, *, model_version, gap):
    """Require DFT for phase minima within a strict gap of the MLIP winner.

    Energies are eV/atom at fixed composition. A DFT relaxed geometry is not
    evidence for the MLIP candidate's single-point ordering. Missing or stale
    data stays unresolved; no calculations are submitted here.
    """
    diagrams = state.get("phase_diagrams") or {}
    snapshot = (state.get("phase_diagrams_by_model") or {}).get(model_version)
    if snapshot is None:
        snapshot = (diagrams.get("mlip_by_version") or {}).get(model_version)
    if snapshot is None:
        snapshot = diagrams.get("mlip") or (diagrams.get("diagrams") or {}).get("mlip")
    if (
        not snapshot
        or not snapshot.get("entries")
        or snapshot.get("status") != "completed"
        or snapshot.get("model_version") != model_version
    ):
        return {
            "status": "missing_evidence",
            "passed": False,
            "reason": "current_model_mlip_phase_snapshot_missing",
        }
    groups = defaultdict(dict)
    invalid = []
    for row in snapshot.get("entries", []):
        try:
            composition = Composition(row["composition"])
            key = _composition_key(row["composition"])
            energy = float(row["normalized_total_energy"]) / composition.num_atoms
            if not row.get("phase") or not row.get("structure_id") or not isfinite(energy):
                raise ValueError("phase/structure identity or finite energy missing")
            item = {**row, "energy_per_atom": energy}
            prior = groups[key].get(row["phase"])
            if prior is None or energy < prior["energy_per_atom"]:
                groups[key][row["phase"]] = item
        except (ValueError, KeyError, TypeError, ZeroDivisionError):
            invalid.append(row.get("structure_id"))
    verified = {row["structure_id"]: row for row in paired}
    checks, missing = [], []
    for composition, phases in groups.items():
        ordered = sorted(phases.values(), key=lambda row: row["energy_per_atom"])
        winner = ordered[0]
        for other in ordered[1:]:
            difference = other["energy_per_atom"] - winner["energy_per_atom"]
            if difference >= gap or isclose(difference, gap, rel_tol=0, abs_tol=1e-12):
                continue
            ids = [winner["structure_id"], other["structure_id"]]
            pair = [verified.get(identity) for identity in ids]
            available = [
                result is not None
                and result.get("phase") == candidate["phase"]
                and abs(result["mlip_energy_eV_per_atom"] - candidate["energy_per_atom"]) <= 1e-8
                and _composition_key(result["composition"]) == composition
                for result, candidate in zip(pair, (winner, other))
            ]
            check = {
                "structure_ids": ids,
                "phases": [winner["phase"], other["phase"]],
                "mlip_gap_ev_per_atom": difference,
            }
            if not all(available):
                check.update(status="missing_same_frame_dft", passed=False)
                missing.extend(identity for identity, valid in zip(ids, available) if not valid)
            else:
                dft_gap = pair[1]["dft_energy_eV_per_atom"] - pair[0]["dft_energy_eV_per_atom"]
                # Ties are unresolved, not a confirmed winner.
                agreement = difference > 0 and dft_gap > 0
                check.update(
                    status="agreement" if agreement else "reversed_or_degenerate",
                    passed=agreement,
                    dft_gap_ev_per_atom=dft_gap,
                )
            checks.append(check)
    passed = not invalid and all(row["passed"] for row in checks)
    return {
        "status": "passed" if passed else "missing_evidence" if missing or invalid else "failed",
        "passed": passed,
        "gap_threshold_ev_per_atom": gap,
        "checks": checks,
        "required_dft_structure_ids": sorted(set(missing)),
        "unidentified_structure_ids": invalid,
        "interpretation": "Current-geometry single-point ordering; larger MLIP gaps are exempt.",
    }


def _composition_key(value):
    composition = Composition(json.loads(value) if isinstance(value, str) else value)
    return tuple(
        sorted(
            (str(el), round(float(count) / composition.num_atoms, 10))
            for el, count in composition.items()
        )
    )

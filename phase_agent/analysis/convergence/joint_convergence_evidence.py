"""Deterministic joint convergence and reliability evidence checklist."""

from __future__ import annotations

import math
from typing import Any


def joint_convergence_checklist(
    state: dict[str, Any],
    *,
    post_dft: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute a compact, zero-LLM scientific convergence checklist."""
    snapshot = (state.get("phase_diagrams") or {}).get("mlip") or {}
    entries = snapshot.get("entries") or []

    competitor_ehulls = [
        float(e["ehull"])
        for e in entries
        if not e.get("is_stable")
        and e.get("ehull") is not None
        and math.isfinite(float(e["ehull"]))
        and float(e["ehull"]) > 1e-6
    ]
    min_margin = min(competitor_ehulls) if competitor_ehulls else None
    near_hull_count = sum(
        1
        for e in entries
        if e.get("ehull") is not None
        and math.isfinite(float(e["ehull"]))
        and 0.0 <= float(e["ehull"]) <= 0.020
    )

    gs_mae = ((post_dft or {}).get("gs_metrics") or {}).get("energy_per_atom", {}).get("mae")
    sample_mae = (
        ((post_dft or {}).get("sample_metrics") or {}).get("energy_per_atom", {}).get("mae")
    )
    ordering_margin_safe = (
        bool(gs_mae < min_margin) if gs_mae is not None and min_margin is not None else None
    )

    qbc_paired = 0
    qbc_blind_spots = 0
    qbc_stds: list[float] = []
    dft_errs: list[float] = []
    for rec in state.get("dft_dataset_records") or []:
        pred = rec.get("mlip_prediction") or {}
        qbc = rec.get("qbc") or {}
        n = int(rec.get("atom_count") or 0)
        e_dft, e_mlip = rec.get("energy"), pred.get("energy")
        e_std = qbc.get("energy_per_atom_std")
        if (
            rec.get("status") == "completed"
            and pred.get("status") == "completed"
            and n > 0
            and e_dft is not None
            and e_mlip is not None
            and isinstance(e_std, (int, float))
            and not isinstance(e_std, bool)
        ):
            err = abs(float(e_mlip) - float(e_dft)) / n
            std_val = float(e_std)
            qbc_paired += 1
            qbc_stds.append(std_val)
            dft_errs.append(err)
            if std_val < 0.005 and err > 0.020:
                qbc_blind_spots += 1

    qbc_error_r = None
    if len(qbc_stds) >= 3:
        mean_s = sum(qbc_stds) / len(qbc_stds)
        mean_e = sum(dft_errs) / len(dft_errs)
        num = sum((s - mean_s) * (e - mean_e) for s, e in zip(qbc_stds, dft_errs))
        den_s = math.sqrt(sum((s - mean_s) ** 2 for s in qbc_stds))
        den_e = math.sqrt(sum((e - mean_e) ** 2 for e in dft_errs))
        if den_s > 1e-12 and den_e > 1e-12:
            qbc_error_r = round(num / (den_s * den_e), 4)

    hotspot_counts: dict[tuple[str, str], int] = {}
    extrapolation_failures = 0
    for t in state.get("tasks") or []:
        output = t.get("outputs") or {}
        failed = (output.get("extrapolation_monitor") or {}).get(
            "extrapolation_failure"
        ) is True or bool(output.get("extrapolation_failures"))
        if not failed:
            continue
        extrapolation_failures += 1
        phase = str(t.get("actual_phase") or t.get("source_phase") or t.get("phase") or "unknown")
        x_na = t.get("x_Na_per_O2")
        na_bin = f"{round(float(x_na), 1):.1f}" if isinstance(x_na, (int, float)) else "unknown"
        key = (phase, na_bin)
        hotspot_counts[key] = hotspot_counts.get(key, 0) + 1
    top_hotspots = [
        {"phase": phase, "x_Na_per_O2_bin": na_bin, "count": count}
        for (phase, na_bin), count in sorted(
            hotspot_counts.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1])
        )[:5]
    ]

    recent_gains: list[int] = []
    for batch in (state.get("relax_mc_batches") or {}).values():
        if not isinstance(batch, dict):
            continue
        rs = batch.get("remote_dft_sampling") or {}
        if rs.get("status") == "accepted":
            recent_gains.append(int(rs.get("gs_count") or 0))
    if not recent_gains:
        for gen in state.get("generation_history") or []:
            if isinstance(gen, dict):
                recent_gains.append(len(gen.get("registered_ids") or []))
    recent_gains = recent_gains[-3:]
    marginal_saturated = (
        bool(len(recent_gains) >= 2 and recent_gains[-1] == 0 and recent_gains[-2] == 0)
        if len(recent_gains) >= 2
        else None
    )

    return {
        "hull_status": snapshot.get("status"),
        "min_competitor_margin_eV_per_atom": (
            round(min_margin, 6) if min_margin is not None else None
        ),
        "gs_energy_mae_eV_per_atom": round(gs_mae, 6) if gs_mae is not None else None,
        "sample_energy_mae_eV_per_atom": (round(sample_mae, 6) if sample_mae is not None else None),
        "ordering_margin_safe": ordering_margin_safe,
        "near_hull_count": near_hull_count,
        "qbc_paired_count": qbc_paired,
        "qbc_blind_spot_count": qbc_blind_spots,
        "qbc_blind_spot_fraction": round(qbc_blind_spots / qbc_paired, 4) if qbc_paired else None,
        "qbc_error_pearson_r": qbc_error_r,
        "extrapolation_failure_count": extrapolation_failures,
        "extrapolation_hotspots": top_hotspots,
        "recent_exploration_gains": recent_gains,
        "marginal_gain_saturated": marginal_saturated,
    }

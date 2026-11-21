"""DFT proposal markers on the same bound MLIP hull as the main figure."""

import json
from hashlib import sha256
from math import isfinite
from pathlib import Path


def plot_dft_selection(state, record, path):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from Process_Phasediagram import plot_phase_diagram
    from phase_agent.analysis.phase.endpoint_reference import endpoint_relative_energies

    from phase_agent.evidence.model_diagram_scope import selection_diagram

    snapshot = selection_diagram(state)
    cycle = state.get("extrapolation_recovery_cycle") or {}
    recovery = (
        cycle.get("status") == "awaiting_reviewed_dft_labels"
        and state.get("active_model_version") == cycle.get("invalid_model_version")
        and snapshot.get("model_version") == cycle.get("baseline_model_version")
        and snapshot.get("version") == cycle.get("baseline_hull_version")
    )
    if snapshot.get("status") != "completed" or (
        snapshot.get("model_version") != state.get("active_model_version") and not recovery
    ):
        raise ValueError("current MLIP hull is unavailable")
    if record.get("model_version") != (
        state.get("active_model_version") if recovery else snapshot["model_version"]
    ):
        raise ValueError("DFT proposal model differs from hull")
    if record.get("hull_version") != snapshot.get("version"):
        raise ValueError("DFT proposal hull differs from current hull")
    entries = snapshot.get("entries") or []
    points = []
    for entry in entries:
        comp = entry.get("composition") or {}
        oxygen, sodium = float(comp.get("O") or 0), float(comp.get("Na") or 0)
        energy = float(entry.get("eform_per_O2", "nan"))
        if oxygen <= 0 or sodium < 0 or not all(map(isfinite, (oxygen, sodium, energy))):
            raise ValueError("invalid phase-diagram coordinate")
        points.append((2 * sodium / oxygen, 1000 * energy))
    if not points:
        raise ValueError("no plottable hull entries")
    relative, _ = endpoint_relative_energies([p[0] for p in points], [p[1] for p in points])
    candidates = {
        r.get("candidate_id") or r.get("structure_id"): r for r in state.get("qbc_candidates") or []
    }
    selected = [
        r
        for r in record.get("decisions") or []
        if r.get("action") in {"DFT_SINGLE_POINT", "DFT_RELAX"}
    ]
    recovery_frames = (
        {r.get("candidate_id"): r.get("structure_sha256") for r in cycle.get("selected") or []}
        if recovery
        else {}
    )
    if recovery and any(r.get("candidate_id") not in recovery_frames for r in selected):
        raise ValueError("recovery_selection_not_bound")
    marked, missing = [], []
    for row in selected:
        cid = row.get("candidate_id")
        candidate = candidates.get(cid) or {}
        digest, path_value = candidate.get("structure_sha256"), candidate.get("structure_path")
        if path_value and digest:
            try:
                if sha256(Path(path_value).read_bytes()).hexdigest() != digest:
                    raise ValueError("candidate frame hash mismatch")
            except OSError:
                digest = None
        indexes = [
            i for i, e in enumerate(entries) if digest and e.get("structure_sha256") == digest
        ]
        if recovery and (not digest or digest != recovery_frames[cid]):
            raise ValueError("recovery_frame_hash_mismatch")
        if recovery or candidate.get("extrapolation_failure") is True:
            indexes = []
        if not indexes:
            if candidate.get("selection_mode") == "weighted_kmeans_pp" and path_value and digest:
                marked_extra = candidate
                comp_c = candidate.get("composition") or {}
                if not comp_c and Path(path_value).is_file():
                    from ase.io import read as ase_read

                    syms = ase_read(path_value, index=0).get_chemical_symbols()
                    comp_c = {"Na": syms.count("Na"), "O": syms.count("O")}
                o_c, na_c = float(comp_c.get("O") or 0), float(comp_c.get("Na") or 0)
                if o_c > 0:
                    x_c = 2 * na_c / o_c
                    ehull_mev = 1000.0 * max(0.0, float(marked_extra.get("ehull") or 0.0))
                    # Place trajectory-sampled point above the nearest hull point at x_c.
                    nearest_idx = min(range(len(points)), key=lambda idx: abs(points[idx][0] - x_c))
                    marked.append(
                        {
                            "candidate_id": cid,
                            "entry_index": None,
                            "xy": (x_c, relative[nearest_idx] + ehull_mev),
                        }
                    )
                    continue
            valid_excluded = bool(
                (recovery or candidate.get("extrapolation_failure") is True)
                and path_value
                and digest
            )
            missing.append(
                {
                    "candidate_id": cid,
                    "reason": (
                        "excluded_recovery_frame" if recovery else "excluded_extrapolation_frame"
                    )
                    if valid_excluded
                    else "not_in_hull_or_frame_unverified",
                    "extrapolation_failure": bool(candidate.get("extrapolation_failure")),
                }
            )
        else:
            marked.extend({"candidate_id": cid, "entry_index": i} for i in indexes)
    fig = Figure(figsize=(6.5, 4.8), constrained_layout=True)
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    plot_phase_diagram(
        [p[0] for p in points],
        relative,
        [e.get("phase") for e in entries],
        [e.get("is_stable") for e in entries],
        phase_hulls=True,
        unstable_alpha=0.65,
        ylabel="Formation energy (meV/O2)",
        ax=ax,
    )
    for pos, item in enumerate(marked):
        if item.get("xy") is not None:
            px, py = item["xy"]
        else:
            i = item["entry_index"]
            px, py = points[i][0], relative[i]
        ax.scatter(
            [px],
            [py],
            facecolors="none",
            edgecolors="#B24A3B",
            linewidths=1.5,
            s=150,
            zorder=20,
            label="DFT selected" if pos == 0 else None,
        )
    excluded_count = sum(
        row["reason"] in {"excluded_extrapolation_frame", "excluded_recovery_frame"}
        for row in missing
    )
    if excluded_count:
        ax.text(
            0.02,
            0.02,
            f"{excluded_count} selected invalid-epoch or extrapolation frames excluded from hull plot",
            transform=ax.transAxes,
            fontsize=8,
            ha="left",
            va="bottom",
        )
    ax.set_title(
        "DFT Selection" + (" | mixed MLIP models" if snapshot.get("mixed_model_versions") else "")
    )
    if marked:
        from matplotlib.lines import Line2D

        existing = ax.get_legend()
        handles = list(existing.legend_handles) if existing else []
        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor="none",
                markeredgecolor="#B24A3B",
                markersize=9,
                label="DFT selected",
            )
        )
        ax.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    info = {
        "hull_version": snapshot.get("version"),
        "model_version": snapshot.get("model_version"),
        "selected_count": len(selected),
        "marked_candidates": sorted({r["candidate_id"] for r in marked}),
        "unplotted_selected": missing,
        "figure_sha256": sha256(path.read_bytes()).hexdigest(),
    }
    metadata = path.with_suffix(".json")
    metadata.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    return {
        "figure": str(path),
        "metadata": str(metadata),
        "unplotted_count": sum(
            row["reason"] not in {"excluded_extrapolation_frame", "excluded_recovery_frame"}
            for row in missing
        ),
        "excluded_extrapolation_count": excluded_count,
    }


def require_dft_selection_plot(state, decisions):
    """Preflight the exact frames before creating DFT tasks or inputs."""
    from tempfile import TemporaryDirectory

    selected = [r for r in decisions if r.get("action") in {"DFT_SINGLE_POINT", "DFT_RELAX"}]
    if not selected:
        return
    from phase_agent.evidence.model_diagram_scope import selection_diagram

    snapshot = selection_diagram(state)
    record = {
        "model_version": state.get("active_model_version"),
        "hull_version": snapshot.get("version"),
        "decisions": selected,
    }
    with TemporaryDirectory() as tmp:
        result = plot_dft_selection(state, record, Path(tmp) / "selection.png")
        if result["unplotted_count"]:
            raise ValueError("dft_selection_plot_unplotted:" + str(result["unplotted_count"]))

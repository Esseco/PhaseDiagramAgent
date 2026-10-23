"""Each panel uses its retained endpoints; influence keeps the original common reference."""

from collections import Counter
from hashlib import sha256
from pathlib import Path

from phase_agent.analysis.phase.phase_quality_preview import build_quality_preview
from phase_agent.analysis.phase.endpoint_reference import (
    endpoint_relative_energies,
    PLOT_RULE_VERSION,
)


def publish_quality_comparison(state, report, *, directory, formats=("png",)):
    if (
        isinstance(formats, str)
        or not formats
        or any(value not in {"png", "pdf", "svg"} for value in formats)
    ):
        raise ValueError("绘图格式仅支持非空的 png/pdf/svg 列表")
    formats = tuple(dict.fromkeys(formats))
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.lines import Line2D
    from Process_Phasediagram import plot_phase_diagram

    snapshot = state["phase_diagrams"]["mlip"]
    excluded = set(report["preview_excluded_geometry_keys"])
    selected = [r for r in report["structures"] if r["geometry_key"] in excluded]
    left = build_quality_preview(snapshot, report)
    try:
        right = build_quality_preview(
            snapshot, report, review_ids=[r["review_id"] for r in selected]
        )
    except ValueError as error:
        return {
            "status": "preview_unavailable",
            "reason": str(error),
            "formal_records_modified": False,
            "plot_paths": [],
        }
    figure = Figure(figsize=(12, 5))
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2, sharey=True)
    figure.subplots_adjust(left=0.075, right=0.83, bottom=0.20, top=0.81, wspace=0.10)
    crossed = {}
    for tier in (1, 2):
        members = [r for r in selected if r["tier"] == tier]
        if members:
            ys, _ = endpoint_relative_energies(
                [r["x"] for r in members],
                [r["energy_mev_per_O2"] for r in members],
                reference=right["endpoint_reference"],
            )
            crossed[tier] = {"x": [r["x"] for r in members], "eform": ys}
    energies = (
        left["plot_data"]["eform"]
        + right["plot_data"]["eform"]
        + [e for group in crossed.values() for e in group["eform"]]
    )
    low, high = min(energies), max(energies)
    padding = max((high - low) * 0.08, 10)
    handles = []
    for index, (ax, data) in enumerate(zip(axes, [left, right])):
        plot_phase_diagram(
            **data["plot_data"],
            phase_hulls=True,
            unstable_alpha=0.65,
            ax=ax,
            legend=index == 0,
            font_size=10,
            ylabel="Formation energy (meV/O2)" if index == 0 else "",
            ylim=(low - padding, high + padding),
        )
        if index == 0:
            handles = ax.get_legend().legend_handles
            labels = [t.get_text() for t in ax.get_legend().get_texts()]
            ax.get_legend().remove()
            before_counts = Counter(left["plot_data"]["phase"])
            after_counts = Counter(right["plot_data"]["phase"])
            for h, label in zip(handles, labels):
                h.set_label(
                    f"{label} (n={before_counts[label]}/{after_counts[label]})"
                    if label in before_counts
                    else label
                )
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_title(
            ("All current structures" if index == 0 else "Hypothetical exclusion for review")
            + f" | n={data['after_count']}",
            fontsize=11,
        )
    for tier, color in [(1, "#B22222"), (2, "#9B6500")]:
        if tier in crossed:
            axes[1].scatter(
                crossed[tier]["x"],
                crossed[tier]["eform"],
                marker="x",
                c=color,
                s=30,
                linewidths=1,
                zorder=0.5,
            )
            handles.append(
                Line2D(
                    [],
                    [],
                    marker="x",
                    linestyle="",
                    color=color,
                    label=f"Tier {tier}: " + ("automatic rule" if tier == 1 else "human review"),
                )
            )
    figure.legend(
        handles=handles, loc="center left", bbox_to_anchor=(0.84, 0.5), frameon=False, fontsize=9
    )
    figure.suptitle(
        "Phase quality review | each panel relative to retained endpoints", fontsize=13, y=0.96
    )
    figure.text(
        0.075,
        0.085,
        f"H>{report['screening_policy']['hull_shift_mev_per_O2']:g}, D>{report['screening_policy']['outlier_gap_mev_per_O2']:g}; direct H>{report['screening_policy']['direct_removal_mev_per_O2']:g} meV/O2. Fixed-reference H: {right['hull_lift_mev_per_O2']:.2f} meV/O2.",
        fontsize=10,
    )
    figure.text(
        0.075,
        0.035,
        "Crosses use the right-panel reference. H is not the vertical gap between panels. "
        + (
            "H/D chain converged."
            if report["recursive_converged"]
            else "INCOMPLETE: chain stopped before convergence."
        ),
        fontsize=9,
    )
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    token = sha256(
        (
            PLOT_RULE_VERSION
            + ":"
            + (
                report.get("input_fingerprint")
                or repr((report["snapshot_evidence_digest"], report["screening_policy"]))
            )
        ).encode()
    ).hexdigest()[:16]
    paths = []
    for extension in formats:
        path = folder / f"phase_review_HD_{token}.{extension}"
        figure.savefig(path, dpi=300)
        paths.append(str(path))
    return {
        "status": "completed",
        "plot_paths": paths,
        "excluded_record_ids": right["excluded_record_ids"],
        "before_count": left["after_count"],
        "after_count": right["after_count"],
        "hull_lift_mev_per_O2": right["hull_lift_mev_per_O2"],
        "formal_records_modified": False,
        "scope": "counterfactual_only_not_rejection",
        "recursive_converged": report["recursive_converged"],
        "recursive_stop_reason": report["recursive_stop_reason"],
        "analysis_passes": len(report["analysis_layers"]),
        "plot_rule_version": PLOT_RULE_VERSION,
        "plot_reference": "each_panel_own_retained_endpoints",
        "influence_reference": "fixed_original",
        "left_endpoint_reference": left["endpoint_reference"],
        "right_endpoint_reference": right["endpoint_reference"],
        "cross_plot_data": crossed,
    }

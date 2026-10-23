"""Round-scoped transition-metal local-moment distributions from saved CSV rows."""

import hashlib
import io
import json
import math
from pathlib import Path


def export_magnetic_plots(rows, directory):
    """Keep magnetic file failures diagnostic so raw DFT recovery still succeeds."""
    try:
        return _export_magnetic_plots(rows, directory)
    except OSError as error:
        return {"status": "unavailable", "reason": str(error), "files": {}}


def _export_magnetic_plots(rows, directory):
    """Plot all recovered finite moments, including rejected results; never infer missing data."""
    from pymatgen.core import Element

    groups, excluded = {}, 0
    for row in rows:
        if not Element(row["element"]).is_transition_metal:
            continue
        try:
            raw = row["moment_mu_B"]
            if row.get("moment_kind") == "vector_xyz":
                vector = json.loads(raw) if isinstance(raw, str) else raw
                if len(vector) != 3:
                    raise ValueError("expected three components")
                value = math.hypot(*map(float, vector))
                kind = "vector norm"
            else:
                value, kind = abs(float(raw)), "absolute scalar"
            if not math.isfinite(value):
                raise ValueError("nonfinite moment")
        except (TypeError, ValueError, OverflowError):
            excluded += 1
            continue
        groups.setdefault((row["element"], kind), []).append((value, row))
    if not groups:
        return {
            "status": "unavailable",
            "reason": "no_finite_transition_metal_moments",
            "excluded_sites": excluded,
            "files": {},
        }
    from phase_agent.analysis.feedback.export_dft_products import _publish_bytes

    directory = Path(directory)
    fingerprint = hashlib.sha256(json.dumps([1, rows], sort_keys=True).encode()).hexdigest()
    manifest = directory / "magnetic_distribution.json"
    if manifest.is_file():
        try:
            saved = json.loads(manifest.read_text(encoding="utf-8"))
            saved["files"] = {
                name: str(directory / Path(path).name) for name, path in saved["files"].items()
            }
            if saved.get("fingerprint") == fingerprint and all(
                Path(path).is_file() for path in saved["files"].values()
            ):
                return saved
        except (ValueError, KeyError, TypeError):
            pass
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError as error:
        return {"status": "unavailable", "reason": str(error), "files": {}}
    files, counts = {}, {}
    try:
        for (element, kind), sites in sorted(groups.items()):
            figure = Figure(figsize=(6.4, 4.2), layout="constrained")
            FigureCanvasAgg(figure)
            axis = figure.subplots()
            values = [value for value, _ in sites]
            axis.hist(
                values,
                bins=min(30, max(1, math.ceil(math.sqrt(len(values))))),
                color="#4477AA",
                edgecolor="white",
            )
            intervals = sorted(
                {
                    (row["expected_abs_min_mu_B"], row["expected_abs_max_mu_B"])
                    for _, row in sites
                    if row.get("expected_abs_min_mu_B") is not None
                    and row.get("expected_abs_max_mu_B") is not None
                }
            )
            if kind == "absolute scalar":
                for low, high in intervals:
                    axis.axvspan(
                        low, high, color="#228833", alpha=0.15, label=f"Allowed: {low:g}-{high:g}"
                    )
                    axis.axvline(low, color="#228833", linestyle="--", linewidth=1)
                    axis.axvline(high, color="#228833", linestyle="--", linewidth=1)
            rejected = sum(row.get("spin_assessment") == "out_of_range" for _, row in sites)
            axis.set(
                title=f"{element}: {len(sites)} sites, {rejected} out of range",
                xlabel=f"Local moment ({kind}, μB)",
                ylabel="Site count",
            )
            axis.yaxis.get_major_locator().set_params(integer=True)
            if intervals and kind == "absolute scalar":
                axis.legend()
            axis.text(
                0.99,
                0.98,
                "All recovered results",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=8,
            )
            suffix = "vector_norm" if kind == "vector norm" else "absolute"
            path = directory / f"{element}_{suffix}_distribution.png"
            stream = io.BytesIO()
            figure.savefig(stream, format="png", dpi=160)
            _publish_bytes(path, stream.getvalue())
            figure.clear()
            files[f"{element}_{suffix}"] = str(path)
            counts[f"{element}_{suffix}"] = len(sites)
        report = {
            "status": "completed",
            "fingerprint": fingerprint,
            "files": files,
            "site_counts": counts,
            "excluded_sites": excluded,
        }
        _publish_bytes(manifest, (json.dumps(report, indent=2) + "\n").encode())
        return report
    except OSError as error:
        return {"status": "unavailable", "reason": str(error), "files": files}

"""Round-scoped transition-metal local-moment distributions from saved CSV rows."""

import hashlib
import io
import json
import math
from pathlib import Path
from statistics import median


def export_magnetic_plots(rows, directory):
    """Keep magnetic file failures diagnostic so raw DFT recovery still succeeds."""
    try:
        return _export_magnetic_plots(rows, directory)
    except (ImportError, OSError) as error:
        return {"status": "unavailable", "reason": str(error), "files": {}}


def _export_magnetic_plots(rows, directory):
    """Plot all recovered finite moments, including rejected results; never infer missing data."""
    from pymatgen.core import Element

    groups, excluded = {}, 0
    for row in rows:
        try:
            if not Element(row["element"]).is_transition_metal:
                continue
            raw = row["moment_mu_B"]
            if isinstance(raw, bool):
                raise ValueError("boolean is not a moment")
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
        except (KeyError, TypeError, ValueError, OverflowError):
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
    fingerprint = hashlib.sha256(json.dumps([2, rows], sort_keys=True).encode()).hexdigest()
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
    files, counts, summaries = {}, {}, {}
    try:
        for (element, kind), sites in sorted(groups.items()):
            figure = Figure(figsize=(6.4, 4.2), layout="constrained")
            FigureCanvasAgg(figure)
            axis = figure.subplots()
            values = [value for value, _ in sites]
            method = _draw_distribution(axis, values)
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
                ylabel="Probability density (μB⁻¹)" if method == "kde" else "",
            )
            if intervals and kind == "absolute scalar":
                axis.legend()
                axis.set_xlim(right=max(axis.get_xlim()[1], max(high for _, high in intervals)))
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
            path = directory / f"{element}_{suffix}_kde.png"
            stream = io.BytesIO()
            figure.savefig(stream, format="png", dpi=160)
            _publish_bytes(path, stream.getvalue())
            figure.clear()
            files[f"{element}_{suffix}"] = str(path)
            key = f"{element}_{suffix}"
            counts[key] = len(sites)
            summaries[key] = {
                "element": element,
                "moment_kind": suffix,
                "site_count": len(sites),
                "task_count": len({row["task_id"] for _, row in sites if row.get("task_id")}),
                "min_mu_B": min(values),
                "median_mu_B": median(values),
                "max_mu_B": max(values),
                "out_of_range_sites": rejected,
                "method": method,
            }
        report = {
            "status": "completed",
            "fingerprint": fingerprint,
            "files": files,
            "site_counts": counts,
            "element_summaries": summaries,
            "excluded_sites": excluded,
        }
        _publish_bytes(manifest, (json.dumps(report, indent=2) + "\n").encode())
        return report
    except OSError as error:
        return {"status": "unavailable", "reason": str(error), "files": files}


def _draw_distribution(axis, values):
    """Reflect the Gaussian density at zero for the nonnegative moment magnitude."""
    import numpy as np
    from scipy.stats import gaussian_kde

    if min(values) != max(values):
        try:
            kernel = gaussian_kde(values)
            bandwidth = float(kernel.cho_cov[0, 0])
            grid = np.linspace(0, max(values) + 5 * bandwidth, 512)
            density = kernel(grid) + kernel(-grid)
            if np.all(np.isfinite(density)):
                axis.plot(grid, density, color="#4477AA")
                axis.fill_between(grid, density, color="#4477AA", alpha=0.2)
                axis.set(xlim=(0, grid[-1]), ylim=(0, None))
                return "kde"
        except (ValueError, np.linalg.LinAlgError):
            pass
    axis.plot(values, [0.5] * len(values), "|", color="#4477AA", markersize=14)
    axis.set(xlim=(0, max(values) * 1.1 or 1), ylim=(0, 1), yticks=[])
    axis.text(0.04, 0.85, "Density unavailable; observed sites shown", transform=axis.transAxes)
    return "observed_sites"

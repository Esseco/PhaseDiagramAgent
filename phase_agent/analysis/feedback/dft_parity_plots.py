"""Round-scoped parity figures from verified pairs, with content-based reuse."""

import hashlib
import io
import json
from pathlib import Path

import numpy as np

PLOT_VERSION = 5


def export_parity_plots(energies, forces, directory, model_label):
    try:
        return _export_parity_plots(energies, forces, directory, model_label)
    except OSError as error:
        return {
            "status": "unavailable",
            "reason": f"对角线图文件发布失败：{error}；数据保留，可在释放文件占用后重试。",
        }


def _export_parity_plots(energies, forces, directory, model_label):
    directory = Path(directory)
    # Local transport paths do not change plotted scientific data.
    content = [
        [{key: value for key, value in row.items() if not key.endswith("_path")} for row in rows]
        for rows in (energies, forces)
    ]
    fingerprint = hashlib.sha256(
        json.dumps([PLOT_VERSION, model_label, *content], sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    manifest = directory / "parity_metrics.json"
    if manifest.is_file():
        try:
            saved = json.loads(manifest.read_text(encoding="utf-8"))
            saved["files"] = {
                key: str(directory / Path(path).name) for key, path in saved["files"].items()
            }
            if (
                saved.get("fingerprint") == fingerprint
                and saved.get("status") == "completed"
                and all(Path(path).is_file() for path in saved["files"].values())
            ):
                return saved
        except (ValueError, KeyError):
            pass  # Corrupt manifest is rebuilt from verified tables, never guessed.
    groups = [
        (
            "energy",
            energies,
            "dft_energy_eV_per_atom",
            "mlip_energy_eV_per_atom",
            "meV/atom",
            "Energy per atom",
        ),
        (
            "force",
            forces,
            "dft_force_eV_per_A",
            "mlip_force_eV_per_A",
            "meV/Å",
            "Atomic force components",
        ),
    ]
    data, metrics = {}, {}
    for name, rows, xkey, ykey, unit, title in groups:
        paired = [r for r in rows if r["comparison_status"] == "completed"]
        if not paired:
            return {"status": "unavailable", "reason": "无完整同帧能量/受力配对，未绘图"}
        x, y = (np.asarray([r[key] for r in paired], dtype=float) for key in (xkey, ykey))
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            return {"status": "unavailable", "reason": "配对数据存在非有限值，未绘图"}
        error = 1000 * (y - x)
        metrics[name] = {
            "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt((error**2).mean())),
            "mae_unit": unit,
            "rmse_unit": unit,
            "points": len(x),
            "structures": len({r["task_id"] for r in paired}),
            "excluded_rows": len(rows) - len(paired),
        }
        data[name] = x, y
    x, y = data["force"]
    dft_rms = float(np.sqrt(np.mean(x * x)))
    mlip_rms = float(np.sqrt(np.mean(y * y)))
    ratio = mlip_rms / dft_rms if dft_rms > 0 else None
    diagnostics = {
        "dft_force_rms": dft_rms,
        "mlip_force_rms": mlip_rms,
        "force_rms_unit": "eV/Å",
        "force_rms_ratio": ratio,
        "warning": "MLIP受力尺度小于DFT的10%；需核查同帧预测接口及模型，不代表已确定原因。"
        if ratio is not None and ratio < 0.1
        else None,
    }
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib import rc_context
    except ImportError as error:
        return {
            "status": "unavailable",
            "reason": f"绘图依赖不可用：{error}",
            "metrics": metrics,
            "diagnostics": diagnostics,
        }
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    with rc_context(
        {"font.family": "sans-serif", "font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"}
    ):
        for name, _, _, _, unit, title in groups:
            x, y = data[name]
            fig = Figure(figsize=(4.8, 5.4))
            FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)
            fig.subplots_adjust(left=0.20, right=0.96, bottom=0.14, top=0.78)
            lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
            pad = max(float(hi - lo) * 0.06, 0.01)
            ax.scatter(
                x,
                y,
                s=22 if name == "energy" else 5,
                alpha=0.8 if name == "energy" else 0.28,
                color="#3679A2",
                linewidths=0,
                rasterized=True,
            )
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="#666666", linewidth=1)
            axis_unit = unit.replace("meV", "eV")
            symbol = "E" if name == "energy" else "F"
            ax.set(
                xlim=(lo - pad, hi + pad),
                ylim=(lo - pad, hi + pad),
                xlabel=rf"${symbol}_{{\mathrm{{DFT}}}}$ ({axis_unit})",
                ylabel=rf"${symbol}_{{\mathrm{{MLIP}}}}$ ({axis_unit})",
            )
            ax.set_aspect("equal", adjustable="box")
            for spine in ax.spines.values():
                spine.set_visible(True)
            m = metrics[name]
            fig.text(0.20, 0.95, title, size=13, weight="bold")
            fig.text(0.20, 0.91, model_label, size=10)
            ax.text(
                0.04,
                0.96,
                f"MAE = {m['mae']:.1f} {unit}\nRMSE = {m['rmse']:.1f} {unit}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=10,
            )
            fig.text(0.20, 0.82, f"n = {m['points']:,}", size=10)
            for extension in ("png", "pdf", "svg"):
                path = directory / f"{name}_parity.{extension}"
                buffer = io.BytesIO()
                fig.savefig(buffer, format=extension, dpi=600)
                from phase_agent.analysis.feedback.export_dft_products import _publish_bytes

                _publish_bytes(path, buffer.getvalue())
                files[f"{name}_{extension}"] = str(path)
            fig.clear()
    report = {
        "status": "completed",
        "fingerprint": fingerprint,
        "files": files,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "averaging": "energy: equal structures; forces: pooled atomic xyz components",
        "selection": "verified completed same-frame pairs only; no outlier removal; not an independent test-set claim",
    }
    temporary = manifest.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(manifest)
    return report

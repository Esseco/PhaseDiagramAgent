"""K-fold out-of-fold MAE trends for registered resulting model epochs."""

import csv
import io
import json
import math
from pathlib import Path

from phase_agent.analysis.feedback.export_dft_products import _publish_bytes
from phase_agent.evidence.model_epoch import model_epoch


def publish_kfold_error_trend(state, root, jobs):
    root = Path(root)
    rows = []
    for job in jobs:
        version = job.get("model_version")
        epoch = model_epoch(state, version) if version else None
        row = {
            "epoch": epoch,
            "source_epoch": job.get("source_epoch"),
            "model_version": version,
            "job_id": job.get("job_id"),
            "fingerprint": job.get("fingerprint"),
            "status": job.get("status"),
            "energy_mae_meV_per_atom": None,
            "force_mae_meV_per_A": None,
            "structures": None,
            "force_components": None,
            "reason": job.get("reason"),
        }
        if row["status"] == "completed":
            try:
                energy, force = job["metrics"]["energy"], job["metrics"]["force"]
                e, f = float(energy["mae"]), float(force["mae"])
                if (
                    energy["mae_unit"] != "meV/atom"
                    or force["mae_unit"] != "meV/Å"
                    or not all(math.isfinite(x) and x >= 0 for x in (e, f))
                ):
                    raise ValueError("invalid MAE or units")
                row.update(
                    energy_mae_meV_per_atom=e,
                    force_mae_meV_per_A=f,
                    structures=energy["structures"],
                    force_components=force["points"],
                )
                if epoch is None:
                    row.update(
                        status="unavailable", reason="resulting model epoch is not registered"
                    )
            except (KeyError, TypeError, ValueError) as error:
                row.update(status="unavailable", reason=str(error))
        rows.append(row)
    rows.sort(
        key=lambda row: (
            int(row["epoch"][5:]) if row["epoch"] else float("inf"),
            str(row["job_id"]),
        )
    )
    latest = root / "latest"
    fields = (
        "epoch",
        "source_epoch",
        "model_version",
        "job_id",
        "fingerprint",
        "status",
        "energy_mae_meV_per_atom",
        "force_mae_meV_per_A",
        "structures",
        "force_components",
        "reason",
    )
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    _publish_bytes(
        latest / "KFold_Error_MAE_by_epoch.csv", csv_buffer.getvalue().encode("utf-8-sig")
    )

    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    for kind, key, ylabel, sample in (
        ("E", "energy_mae_meV_per_atom", "Energy MAE (meV/atom)", "structures"),
        ("F", "force_mae_meV_per_A", "Force MAE (meV/Å)", "force_components"),
    ):
        figure = Figure(figsize=(7, 4.8))
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        valid = [row for row in rows if row["status"] == "completed" and row["epoch"]]
        xs = [int(row["epoch"][5:]) for row in valid]
        ys = [row[key] for row in valid]
        if valid:
            # Multiple training jobs can share an epoch; preserve each point.
            if len(xs) == len(set(xs)) and all(b == a + 1 for a, b in zip(xs, xs[1:])):
                axis.plot(xs, ys, marker="o")
            else:
                axis.scatter(xs, ys)
            for x, y, row in zip(xs, ys, valid):
                axis.annotate(
                    f"n={row[sample]}",
                    (x, y),
                    xytext=(4, 6),
                    textcoords="offset points",
                    fontsize=8,
                )
        else:
            axis.text(
                0.5,
                0.5,
                "No verified K-fold results with a registered model epoch",
                transform=axis.transAxes,
                ha="center",
                va="center",
            )
        epochs = sorted({int(row["epoch"][5:]) for row in rows if row["epoch"]})
        axis.set_xticks(epochs, [f"epoch{x}" for x in epochs])
        axis.set(xlabel="Resulting model epoch", ylabel=ylabel, title="K-fold out-of-fold MAE")
        figure.text(
            0.5, 0.02, "Training sets may differ; not an independent test", ha="center", fontsize=8
        )
        figure.tight_layout(rect=(0, 0.06, 1, 1))
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=200)
        _publish_bytes(latest / f"KFold_Error_{kind}_MAE_by_epoch.png", buffer.getvalue())
        figure.clear()
    report = {
        "evaluation_type": "kfold_out_of_fold",
        "interpretation": "Epoch identifies the registered resulting model; source_epoch identifies its training source. Training datasets may differ; not an independent test.",
        "rows": rows,
    }
    _publish_bytes(
        latest / "KFold_Error_MAE_by_epoch.json",
        (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    return report

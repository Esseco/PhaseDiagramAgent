"""Build a piecewise average Na voltage curve from a verified composition hull."""

import csv
import json
from math import isfinite
from pathlib import Path


VOLTAGE_RULE_VERSION = "hull-segment-average-v2"
DEFAULT_NA_REFERENCE_EVPERATOM = -1.288
_FIELDS = (
    "status",
    "reason",
    "model_version",
    "hull_version",
    "x_start_Na_per_O2",
    "x_end_Na_per_O2",
    "x_mid_Na_per_O2",
    "delta_x_Na_per_O2",
    "hull_energy_start_eV_per_O2",
    "hull_energy_end_eV_per_O2",
    "chemical_potential_slope_eV_per_Na",
    "reference_energy_eV_per_Na",
    "voltage_V",
    "start_record_id",
    "end_record_id",
    "start_phase",
    "end_phase",
)


def voltage_reference(state):
    conventions = ((state.get("confirmed_config") or {}).get("energy_conventions") or {})
    row = (conventions.get("voltage_references") or {}).get("Na")
    if not isinstance(row, dict):
        return {
            "value_eV_per_Na": DEFAULT_NA_REFERENCE_EVPERATOM,
            "source": "program_default:Na=-1.288 eV/atom",
            "defaulted": True,
        }, None
    value = row.get("value")
    unit = row.get("unit", "eV/atom")
    source = row.get("source")
    if not isinstance(value, (int, float)) or not isfinite(float(value)):
        return None, "voltage_reference_invalid_value"
    if unit != "eV/atom":
        return None, "voltage_reference_unit_must_be_eV_per_atom"
    if not isinstance(source, str) or not source.strip():
        return None, "voltage_reference_source_missing"
    return {"value_eV_per_Na": float(value), "source": source, "defaulted": False}, None


def _status(snapshot, reason, *, reference=None, rows=()):
    return {
        "status": "completed" if reason is None else "not_ready",
        "reason": reason,
        "rule_version": VOLTAGE_RULE_VERSION,
        "model_version": snapshot.get("model_version"),
        "hull_version": snapshot.get("version"),
        "reference": reference,
        "segment_count": len(rows),
    }


def build_voltage_curve(snapshot, state):
    """Return status and one row per stable hull segment, without mutating inputs."""
    reference, reason = voltage_reference(state)
    if reason:
        return _status(snapshot, reason, reference=reference), []
    if snapshot.get("status") != "completed":
        return _status(snapshot, "phase_hull_not_completed", reference=reference), []

    points = {}
    for entry in snapshot.get("entries") or []:
        if not entry.get("is_stable"):
            continue
        x = entry.get("x_Na_per_O2")
        energy = entry.get("hull_energy_per_O2", entry.get("energy_per_O2"))
        if not isinstance(x, (int, float)) or not isinstance(energy, (int, float)):
            continue
        if not isfinite(float(x)) or not isfinite(float(energy)):
            continue
        key = round(float(x), 10)
        candidate = (float(energy), str(entry.get("record_id") or ""), entry)
        previous = points.get(key)
        if previous is None or candidate[:2] < previous[:2]:
            points[key] = candidate
    ordered = [(x, value[0], value[2]) for x, value in sorted(points.items())]
    if len(ordered) < 2:
        return _status(snapshot, "at_least_two_stable_compositions_required", reference=reference), []

    rows = []
    for (x0, e0, start), (x1, e1, end) in zip(ordered, ordered[1:]):
        delta = x1 - x0
        if delta <= 0:
            continue
        slope = (e1 - e0) / delta
        rows.append(
            {
                "status": "completed",
                "reason": None,
                "model_version": snapshot.get("model_version"),
                "hull_version": snapshot.get("version"),
                "x_start_Na_per_O2": x0,
                "x_end_Na_per_O2": x1,
                "x_mid_Na_per_O2": (x0 + x1) / 2,
                "delta_x_Na_per_O2": delta,
                "hull_energy_start_eV_per_O2": e0,
                "hull_energy_end_eV_per_O2": e1,
                "chemical_potential_slope_eV_per_Na": slope,
                "reference_energy_eV_per_Na": reference["value_eV_per_Na"],
                "voltage_V": -(slope - reference["value_eV_per_Na"]),
                "start_record_id": start.get("record_id"),
                "end_record_id": end.get("record_id"),
                "start_phase": start.get("phase"),
                "end_phase": end.get("phase"),
            }
        )
    if not rows:
        return _status(snapshot, "no_positive_hull_segments", reference=reference), []
    return _status(snapshot, None, reference=reference, rows=rows), rows


def write_voltage_outputs(snapshot, state, png_path, csv_path, status_path):
    """Write a status, CSV and plot as one derived publication set."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    status, rows = build_voltage_curve(snapshot, state)
    csv_target, png_target, status_target = map(Path, (csv_path, png_path, status_path))
    for target in (csv_target, png_target, status_target):
        target.parent.mkdir(parents=True, exist_ok=True)

    with csv_target.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        if not rows:
            writer.writerow({"status": status["status"], "reason": status.get("reason")})

    status_target.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fig = Figure(figsize=(7.0, 4.5), constrained_layout=True)
    FigureCanvasAgg(fig)
    axis = fig.subplots()
    if rows:
        # Draw one connected step curve: horizontal voltage plateaus joined by
        # vertical transitions at composition boundaries.  Keep the figure
        # uncluttered; exact values remain in the CSV and status metadata.
        x_values = [rows[0]["x_start_Na_per_O2"]]
        voltage_values = [rows[0]["voltage_V"]]
        for index, row in enumerate(rows):
            x_values.append(row["x_end_Na_per_O2"])
            voltage_values.append(row["voltage_V"])
            if index + 1 < len(rows):
                x_values.append(row["x_end_Na_per_O2"])
                voltage_values.append(rows[index + 1]["voltage_V"])
        axis.plot(x_values, voltage_values, linewidth=2.5, color="#245a9b")
        axis.set_ylabel("Average voltage (V)")
        axis.set_xlabel("Na content (Na/O2)")
        axis.grid(True, alpha=0.25)
        axis.set_title(f"Na voltage plateaus | {snapshot.get('model_version')}")
    else:
        axis.axis("off")
        axis.text(0.5, 0.5, "Voltage curve not ready\\n" + str(status.get("reason")), ha="center", va="center")
    fig.savefig(png_target, dpi=220)
    return {**status, "csv_path": str(csv_target), "plot_path": str(png_target), "status_path": str(status_target)}

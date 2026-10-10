"""Conservative DFT-basis hull; never combine raw MLIP and DFT energies."""

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from phase_agent.analysis.feedback.dft_comparison_tables import build_comparison_tables
from phase_agent.analysis.phase.build_composition_hull_entries import build_composition_hull_entries
from phase_agent.analysis.phase.export_phase_diagram_csv import export_phase_diagram_csv
from phase_agent.analysis.phase.phase_snapshot_paths import model_output_directory
from phase_agent.analysis.phase.publish_current_csv import publish_current_csv

SOURCE_FIELDS = (
    "mlip_energy_eV",
    "dft_energy_eV",
    "corrected_energy_eV",
    "energy_source",
    "correction_status",
    "correction_eV_per_O2",
)


def _coordinates(row):
    comp = row["composition"]
    units = float(comp.get("O", 0)) / 2
    if units <= 0:
        raise ValueError("综合相图要求含氧结构")
    framework = tuple(
        sorted(
            (key, round(float(value) / units, 9))
            for key, value in comp.items()
            if key not in {"Na", "O"}
        )
    )
    return units, float(comp.get("Na", 0)) / units, framework


def build_combined_phase_diagram(mlip_snapshot, records, comparisons):
    version = mlip_snapshot.get("model_version")
    if not version:
        raise ValueError("综合相图需要明确原模型版本")
    records = [r for r in records if r.get("model_version") == version]
    energies, _ = build_comparison_tables(records, comparisons, round_name="calibration")
    paired = {r["task_id"]: r for r in energies if r["comparison_status"] == "completed"}
    anchors, direct = defaultdict(lambda: defaultdict(list)), {}
    rows = []
    for record in records:
        pair = paired.get(record["task_id"])
        phase = record.get("actual_phase")
        identification = record.get("phase_identification") or {}
        if pair is None or not phase or identification.get("status") != "identified":
            continue
        units, x, framework = _coordinates(record)
        delta = (pair["dft_energy_eV"] - pair["mlip_energy_eV"]) / units
        anchors[(phase, framework)][x].append(delta)
        row = {
            **deepcopy(record),
            "record_id": record["task_id"],
            "phase": phase,
            "phase_identification_status": "identified",
            "structure_sha256": record.get("structure_sha256"),
            "original_energy": pair["mlip_energy_eV"],
            "normalized_total_energy": pair["dft_energy_eV"],
            "mlip_energy_eV": pair["mlip_energy_eV"],
            "dft_energy_eV": pair["dft_energy_eV"],
            "corrected_energy_eV": pair["dft_energy_eV"],
            "energy_source": "dft",
            "correction_status": "verified_same_frame",
            "correction_eV_per_O2": delta,
        }
        rows.append(row)
        # Only identical geometry can replace a MLIP row, never structure_id alone.
        if row.get("structure_sha256"):
            direct[(row["structure_sha256"], phase, framework)] = row
    for original in mlip_snapshot.get("entries") or []:
        row = deepcopy(original)
        units, x, framework = _coordinates(row)
        phase = row.get("phase")
        if (row.get("structure_sha256"), phase, framework) in direct:
            continue
        energy = float(row["normalized_total_energy"])
        if not math.isfinite(energy):
            raise ValueError("综合相图输入能量非有限值")
        row.update(
            mlip_energy_eV=energy,
            dft_energy_eV=None,
            corrected_energy_eV=None,
            energy_source="unavailable",
            correction_status="insufficient_calibration",
            correction_eV_per_O2=None,
        )
        points = anchors.get((phase, framework), {})
        xs = sorted(points)
        if (
            row.get("phase_identification_status") == "identified"
            and len(xs) >= 2
            and xs[0] <= x <= xs[-1]
        ):
            delta = float(np.interp(x, xs, [np.mean(points[key]) for key in xs]))
            row.update(
                normalized_total_energy=energy + delta * units,
                corrected_energy_eV=energy + delta * units,
                energy_source="corrected_mlip",
                correction_status="interpolated_same_phase",
                correction_eV_per_O2=delta,
            )
        rows.append(row)
    # Reject different TM systems even if their calibration is incomplete.
    if len({_coordinates(row)[2] for row in rows}) > 1:
        raise ValueError("检测到 TM/O2 组分不一致；综合相图只允许 Na 变化")
    usable = [r for r in rows if r["energy_source"] != "unavailable"]
    snapshot = {
        "method": "combined",
        "model_version": version,
        "energy_basis_id": f"dft_corrected:{version}:per_O2",
        "correction_method": "same_phase_Na_linear_interpolation_no_extrapolation_v1",
        "entries": [],
        "status": "unknown",
        "excluded_structures": len(rows) - len(usable),
    }
    try:
        hull = build_composition_hull_entries(usable)
        for entry, row in zip(hull, usable, strict=True):
            entry.update({key: row.get(key) for key in SOURCE_FIELDS})
        snapshot.update(
            entries=hull,
            status="completed" if len(rows) == len(usable) else "partial",
            hull_domain="calibrated_observed_Na_endpoints",
        )
    except ValueError as error:
        snapshot["reason"] = str(error)
    for row in rows:
        if row["energy_source"] == "unavailable" or snapshot["status"] == "unknown":
            # No raw-MLIP Eform/Ehull may masquerade as a calibrated result.
            snapshot["entries"].append(
                {
                    key: row.get(key)
                    for key in (
                        "record_id",
                        "structure_id",
                        "structure_path",
                        "structure_sha256",
                        "structure",
                        "phase",
                        "phase_identification_status",
                        "composition",
                        "original_energy",
                        *SOURCE_FIELDS,
                    )
                }
            )
    snapshot["version"] = hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()[
        :12
    ]
    return snapshot


def refresh_combined_phase_diagram(state, directory):
    """Agent-side publishing; no model inference, training or task submission."""
    mlip = (state.get("phase_diagrams") or {}).get("mlip") or {}
    if mlip.get("status") != "completed" or not state.get("dft_dataset_records"):
        return
    try:
        phases = {row.get("source_task_id"): row for row in state.get("phase_records") or []}
        records = []
        for original in state.get("dft_dataset_records") or []:
            row = deepcopy(original)
            evidence = phases.get(row.get("task_id")) or {}
            if not row.get("structure_sha256"):
                row["structure_sha256"] = evidence.get("structure_sha256")
            records.append(row)
        snapshot = build_combined_phase_diagram(
            mlip, records, state.get("dft_mlip_comparisons") or []
        )
    except ValueError as error:
        state.setdefault("phase_diagrams", {})["combined"] = {
            "method": "combined",
            "status": "failed",
            "reason": str(error),
            "entries": [],
        }
        return
    if directory is not None:
        root = (
            model_output_directory(directory, snapshot["model_version"], state=state)
            / "phase_diagrams"
            / "combined"
        )
        from phase_agent.analysis.state.model_epoch import model_epoch

        snapshot["epoch"] = model_epoch(state, snapshot["model_version"])
        history = root / "history"
        history.mkdir(parents=True, exist_ok=True)
        target = history / f"phase_diagram_combined_{snapshot['version']}.csv"
        if not target.is_file():
            export_phase_diagram_csv(snapshot, target)
        publish_current_csv(snapshot, target)
        metadata = target.with_suffix(".json")
        if not metadata.is_file():
            metadata.write_text(
                json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
    state.setdefault("phase_diagrams", {})["combined"] = snapshot
    from phase_agent.analysis.feedback.export_dft_products import publish_output_catalog

    publish_output_catalog(state, directory)

"""Export one immutable phase diagram snapshot as a flat CSV table."""

from __future__ import annotations

import csv
import json
from pathlib import Path


FIELDS = (
    "hull_version", "energy_method", "energy_basis_id", "source_version",
    "record_id", "structure_id", "phase", "phase_identification_status", "structure_sha256",
    "x_Na_per_O2", "composition", "is_composition_ground_state",
    "endpoint_x_min", "endpoint_x_max", "endpoint_min_energy_eV_per_O2",
    "endpoint_max_energy_eV_per_O2", "eform_eV_per_O2", "hull_eform_eV_per_O2",
    "original_energy", "normalized_total_energy_eV", "ehull_eV_per_atom",
    "is_stable", "structure_path", "na_layer_uniform", "na_layer_status",
    "na_layer_rule", "model_version", "energy_eV_per_O2", "hull_energy_eV_per_O2", "ehull_eV_per_O2",
)


def export_phase_diagram_csv(snapshot: dict, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for entry in snapshot.get("entries") or []:
            composition = entry.get("composition") or {}
            oxygen = composition.get("O") or 0
            sodium = composition.get("Na") or 0
            x = format(2 * float(sodium) / float(oxygen), ".10f") if oxygen else ""
            writer.writerow({
                "hull_version": snapshot.get("version"),
                "energy_method": snapshot.get("method"),
                "energy_basis_id": snapshot.get("energy_basis_id"),
                "source_version": entry.get("source_version"),
                "record_id": entry.get("record_id"),
                "structure_id": entry.get("structure_id"),
                "phase": entry.get("phase"),
                "phase_identification_status": entry.get("phase_identification_status"),
                "structure_sha256": entry.get("structure_sha256"),
                "x_Na_per_O2": x,
                "composition": json.dumps(composition, ensure_ascii=False, sort_keys=True),
                "is_composition_ground_state": entry.get("is_composition_ground_state"),
                "endpoint_x_min": entry.get("endpoint_x_min"),
                "endpoint_x_max": entry.get("endpoint_x_max"),
                "endpoint_min_energy_eV_per_O2": entry.get("endpoint_min_energy_per_O2"),
                "endpoint_max_energy_eV_per_O2": entry.get("endpoint_max_energy_per_O2"),
                "eform_eV_per_O2": entry.get("eform_per_O2"),
                "hull_eform_eV_per_O2": entry.get("hull_eform_per_O2"),
                "original_energy": entry.get("original_energy"),
                "normalized_total_energy_eV": entry.get("normalized_total_energy"),
                "ehull_eV_per_atom": entry.get("ehull"),
                "is_stable": entry.get("is_stable"),
                "structure_path": entry.get("structure_path"),
                "na_layer_uniform": entry.get("na_layer_uniform"),
                "na_layer_status": entry.get("na_layer_status"),
                "na_layer_rule": entry.get("na_layer_rule"),
                "model_version": snapshot.get("model_version"),
                "energy_eV_per_O2": entry.get("energy_per_O2"),
                "hull_energy_eV_per_O2": entry.get("hull_energy_per_O2"),
                "ehull_eV_per_O2": entry.get("ehull_per_O2"),
            })
    temporary.replace(target)
    return target

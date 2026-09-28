"""Export one immutable phase diagram snapshot as a flat CSV table."""

from __future__ import annotations

import csv
import json
from fractions import Fraction
from pathlib import Path


FIELDS = (
    "hull_version", "energy_method", "energy_basis_id", "source_version",
    "record_id", "structure_id", "phase", "x_Na_per_O2", "composition",
    "original_energy", "normalized_total_energy_eV", "ehull_eV_per_atom",
    "is_stable", "structure_path", "na_layer_uniform", "na_layer_status",
    "na_layer_rule",
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
            x = str(Fraction(str(2 * sodium)) / Fraction(str(oxygen))) if oxygen else ""
            writer.writerow({
                "hull_version": snapshot.get("version"),
                "energy_method": snapshot.get("method"),
                "energy_basis_id": snapshot.get("energy_basis_id"),
                "source_version": entry.get("source_version"),
                "record_id": entry.get("record_id"),
                "structure_id": entry.get("structure_id"),
                "phase": entry.get("phase"),
                "x_Na_per_O2": x,
                "composition": json.dumps(composition, ensure_ascii=False, sort_keys=True),
                "original_energy": entry.get("original_energy"),
                "normalized_total_energy_eV": entry.get("normalized_total_energy"),
                "ehull_eV_per_atom": entry.get("ehull"),
                "is_stable": entry.get("is_stable"),
                "structure_path": entry.get("structure_path"),
                "na_layer_uniform": entry.get("na_layer_uniform"),
                "na_layer_status": entry.get("na_layer_status"),
                "na_layer_rule": entry.get("na_layer_rule"),
            })
    temporary.replace(target)
    return target

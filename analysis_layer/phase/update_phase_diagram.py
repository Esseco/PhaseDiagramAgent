"""按统一能量口径分别更新 MLIP 与 DFT 凸包。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition

from analysis_layer.phase.check_na_layer_uniformity import check_na_layer_uniformity
from analysis_layer.phase.export_phase_diagram_csv import export_phase_diagram_csv


def update_phase_diagram(
    records: list[dict[str, Any]],
    *,
    energy_basis: str = "total",
    output_directory: str | Path | None = None,
    parent_versions: dict[str, str] | None = None,
) -> dict[str, Any]:
    """分别构建 mlip/dft 相图，保留原能量并保存可追踪版本。"""
    if energy_basis not in {"total", "per_atom", "per_formula_unit"}:
        raise ValueError("energy_basis 必须是 total/per_atom/per_formula_unit")
    output = {"energy_basis": energy_basis, "diagrams": {}}
    for method in ("mlip", "dft"):
        selected = [
            item
            for item in records
            if str(item.get("energy_method", "")).lower() == method
            and item.get("status", "completed") == "completed"
        ]
        normalized, rejected = [], []
        for item in selected:
            try:
                if item.get("energy_unit") != "eV":
                    raise ValueError("相图记录 energy_unit 必须为 'eV'")
                composition = Composition(item["composition"])
                energy = _total_energy(item, composition, energy_basis)
                normalized.append(
                    {
                        **item,
                        "original_energy": item["energy"],
                        "normalized_total_energy": energy,
                        "composition": composition.as_dict(),
                    }
                )
            except (KeyError, TypeError, ValueError) as error:
                rejected.append(
                    {"record_id": item.get("record_id"), "reason": str(error)}
                )
        snapshot = _build_snapshot(
            method,
            normalized,
            rejected,
            energy_basis,
            (parent_versions or {}).get(method),
        )
        output["diagrams"][method] = snapshot
        if output_directory is not None:
            directory = Path(output_directory)
            directory.mkdir(parents=True, exist_ok=True)
            csv_path = directory / f"phase_diagram_{method}_{snapshot['version']}.csv"
            export_phase_diagram_csv(snapshot, csv_path)
            snapshot["csv_path"] = str(csv_path)
            path = directory / f"phase_diagram_{method}_{snapshot['version']}.json"
            path.write_text(
                json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            snapshot["path"] = str(path)
    return output


def _total_energy(item, composition, basis):
    energy = float(item["energy"])
    if basis == "per_atom":
        return energy * composition.num_atoms
    if basis == "per_formula_unit":
        units = item.get("formula_units")
        if not isinstance(units, (int, float)) or units <= 0:
            raise ValueError("per_formula_unit 需要正数 formula_units")
        return energy * units
    return energy


def _build_snapshot(method, records, rejected, basis, parent):
    payload = [
        {
            "record_id": item.get("record_id"),
            "composition": item["composition"],
            "energy": item["normalized_total_energy"],
        }
        for item in records
    ]
    version = hashlib.sha256(
        json.dumps(
            {"method": method, "basis": basis, "entries": payload}, sort_keys=True
        ).encode()
    ).hexdigest()[:12]
    snapshot = {
        "method": method,
        "energy_basis": basis,
        "energy_basis_id": f"{method}:{basis}",
        "version": version,
        "parent_version": parent,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "entries": [],
        "rejected": rejected,
        "status": "completed",
    }
    try:
        entries = [
            PDEntry(
                item["composition"],
                item["normalized_total_energy"],
                name=item.get("record_id"),
            )
            for item in records
        ]
        diagram = PhaseDiagram(entries)
        for item, entry in zip(records, entries, strict=True):
            na_check = check_na_layer_uniformity(
                item.get("structure_path"), item["composition"])
            snapshot["entries"].append(
                {
                    "record_id": item.get("record_id"),
                    "structure_id": item.get("structure_id"),
                    "structure_path": item.get("structure_path"),
                    "phase": item.get("phase"),
                    "composition": item["composition"],
                    "original_energy": item["original_energy"],
                    "normalized_total_energy": item["normalized_total_energy"],
                    "ehull": float(diagram.get_e_above_hull(entry)),
                    "ehull_unit": "eV/atom",
                    "is_stable": entry in diagram.stable_entries,
                    "source_version": item.get("source_version"),
                    "na_layer_uniform": na_check["uniform"],
                    "na_layer_status": na_check["status"],
                    "na_layer_rule": na_check["rule"],
                }
            )
    except Exception as error:
        snapshot.update(
            {"status": "failed", "error": f"{type(error).__name__}: {error}"}
        )
    return snapshot

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
    active_model_version: str | None = None,
) -> dict[str, Any]:
    """分别构建 mlip/dft 相图，保留原能量并保存可追踪版本。"""
    if energy_basis not in {"total", "per_atom", "per_formula_unit"}:
        raise ValueError("energy_basis 必须是 total/per_atom/per_formula_unit")
    output = {"energy_basis": energy_basis, "diagrams": {}, "mlip_by_version": {}}
    versions = sorted({str(item.get("model_version") or item.get("source_version") or "unknown")
                       for item in records if item.get("energy_method") == "mlip"})
    groups = [("mlip", version) for version in versions] + [("dft", None)]
    for method, model_version in groups:
        selected = [
            item
            for item in records
            if str(item.get("energy_method", "")).lower() == method
            and item.get("status", "completed") == "completed"
            and (method != "mlip" or str(item.get("model_version") or item.get("source_version") or "unknown") == model_version)
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
        if method == "mlip":
            snapshot["model_version"] = model_version
            snapshot["energy_basis_id"] = f"mlip:{model_version}:{energy_basis}"
            snapshot["version"] = hashlib.sha256(
                f"{model_version}:{snapshot['version']}".encode()).hexdigest()[:12]
            snapshot["parent_version"] = (parent_versions or {}).get(f"mlip:{model_version}")
            output["mlip_by_version"][model_version] = snapshot
        else:
            output["diagrams"][method] = snapshot
        if output_directory is not None:
            directory = Path(output_directory)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"phase_diagram_{method}_{snapshot['version']}.json"
            if path.is_file():
                # The same input must reuse the original immutable snapshot.
                stored = json.loads(path.read_text(encoding="utf-8"))
                if stored.get("version") == snapshot["version"]:
                    stored["path"] = str(path)
                    snapshot.clear()
                    snapshot.update(stored)
                    continue
            csv_path = directory / f"phase_diagram_{method}_{snapshot['version']}.csv"
            export_phase_diagram_csv(snapshot, csv_path)
            snapshot["csv_path"] = str(csv_path)
            snapshot["path"] = str(path)
            path.write_text(
                json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
    chosen = active_model_version or (versions[0] if len(versions) == 1 else None)
    output["diagrams"]["mlip"] = output["mlip_by_version"].get(chosen) or {
        "method": "mlip", "model_version": chosen, "status": "unknown",
        "reason": "active_model_version_required" if chosen is None else "no_results_for_active_model",
        "entries": []}
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
            "structure_id": item.get("structure_id"),
            "structure_path": item.get("structure_path"),
            "structure_sha256": item.get("structure_sha256"),
            "phase": item.get("phase"),
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
    if not records:
        snapshot.update({"status": "unknown", "reason": "no_identified_energy_records"})
        return snapshot
    try:
        elements = {element for item in records for element in item["composition"]}
        elemental_endpoints = {next(iter(item["composition"])) for item in records
                               if len(item["composition"]) == 1}
        if records and not elements.issubset(elemental_endpoints):
            from analysis_layer.phase.build_composition_hull_entries import build_composition_hull_entries
            snapshot["entries"] = build_composition_hull_entries(records)
            snapshot["hull_domain"] = "covered_compositions_only"
            return snapshot
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

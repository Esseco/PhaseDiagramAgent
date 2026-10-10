"""Validate MLIP/DFT static energies on identical final geometries by branch."""

from __future__ import annotations


def validate_branch_static_energies(records, *, model_version, energy_unit="eV/atom"):
    groups = {}
    rejected = []
    for row in records:
        required = (
            row.get("branch_id"),
            row.get("geometry_id"),
            row.get("mlip_energy"),
            row.get("dft_energy"),
        )
        if any(value is None for value in required):
            rejected.append(
                {"record_id": row.get("record_id"), "reason": "missing_same_geometry_evidence"}
            )
            continue
        if row.get("model_version") != model_version:
            rejected.append({"record_id": row.get("record_id"), "reason": "model_version_mismatch"})
            continue
        if (
            row.get("mlip_geometry_id", row["geometry_id"]) != row["geometry_id"]
            or row.get("dft_geometry_id", row["geometry_id"]) != row["geometry_id"]
        ):
            rejected.append({"record_id": row.get("record_id"), "reason": "geometry_mismatch"})
            continue
        if row.get("energy_unit", energy_unit) != energy_unit:
            rejected.append({"record_id": row.get("record_id"), "reason": "energy_unit_mismatch"})
            continue
        groups.setdefault(row["branch_id"], []).append(
            abs(float(row["mlip_energy"]) - float(row["dft_energy"]))
        )
    by_branch = {
        branch: {"count": len(values), "mae_ev_per_atom": sum(values) / len(values)}
        for branch, values in sorted(groups.items())
    }
    all_values = [value for values in groups.values() for value in values]
    return {
        "status": "completed" if all_values else "insufficient_data",
        "model_version": model_version,
        "energy_unit": energy_unit,
        "by_branch": by_branch,
        "overall_mae_ev_per_atom": sum(all_values) / len(all_values) if all_values else None,
        "accepted_count": len(all_values),
        "rejected": rejected,
    }

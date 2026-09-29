"""Use the existing composition-constrained convex envelope for electrode data."""
from pymatgen.core import Composition

from analysis_layer.phase.branch_relax_hull import hull_energy_per_atom
from analysis_layer.phase.check_na_layer_uniformity import check_na_layer_uniformity


def build_composition_hull_entries(records):
    pool = {"records": [{**row, "energy": row["normalized_total_energy"]} for row in records]}
    entries = []
    for row in records:
        composition = Composition(row["composition"])
        reference = hull_energy_per_atom(pool, row["composition"])
        if reference is None:
            raise ValueError("covered composition has no feasible hull reference")
        energy = row["normalized_total_energy"] / composition.num_atoms
        gap = energy - reference
        check = check_na_layer_uniformity(row.get("structure_path"), row["composition"])
        units = float(composition["O"]) / 2
        entries.append({"record_id": row.get("record_id"), "structure_id": row.get("structure_id"),
            "structure_path": row.get("structure_path"), "phase": row.get("phase"),
            "composition": row["composition"], "original_energy": row["original_energy"],
            "normalized_total_energy": row["normalized_total_energy"],
            "ehull": gap, "ehull_unit": "eV/atom", "is_stable": abs(gap) <= 1e-8,
            "energy_per_O2": row["normalized_total_energy"] / units if units else None,
            "hull_energy_per_O2": reference * composition.num_atoms / units if units else None,
            "ehull_per_O2": gap * composition.num_atoms / units if units else None,
            "source_version": row.get("source_version"),
            "na_layer_uniform": check["uniform"], "na_layer_status": check["status"],
            "na_layer_rule": check["rule"]})
    return entries

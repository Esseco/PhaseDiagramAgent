"""Rank Na/vacancy orderings before removing any candidate Na site."""

from __future__ import annotations

from fractions import Fraction

from pymatgen.transformations.standard_transformations import OrderDisorderedStructureTransformation

from phase_agent.science.structures.generate_branch_structure import (
    _assign_charge_balanced_average_tm_valence,
)


def rank_na_orderings_by_electrostatics(full, x, *, limit=10, oxidation_states=None):
    """Return ``(structure, Ewald score, charge scheme, raw rank)`` in score order.

    Scores are ranking diagnostics, not MLIP/DFT energies.  A missing score or
    failed charge assignment is an error, never a reason to silently use a
    random ordering.
    """
    na_sites = [i for i, site in enumerate(full) if site.is_ordered and site.specie.symbol == "Na"]
    oxygen = sum(site.is_ordered and site.specie.symbol == "O" for site in full)
    target = Fraction(str(x)) * oxygen / 2
    if target.denominator != 1 or not 0 <= target <= len(na_sites):
        raise ValueError(f"x={x} requires an invalid number of Na sites: {target}")
    if target == 0 or target == len(na_sites):
        result = full.copy()
        if target == 0:
            result.remove_species(["Na"])
        return [(result, None, "endpoint_no_ranking", 0)]
    disordered = full.copy()
    occupancy = int(target) / len(na_sites)
    for index in na_sites:
        disordered.replace(index, {"Na": occupancy})
    layer_groups = _na_layer_groups(full, na_sites)
    if len(layer_groups) > int(target):
        raise ValueError(
            f"x={x} 只有 {int(target)} 个 Na，少于 {len(layer_groups)} 个 Na 层；"
            "无法满足每层至少一个 Na"
        )
    try:
        ranked = _rank(disordered, full, x, limit, oxidation_states, layer_groups, na_sites)
        scheme = "provided_oxidation_states" if oxidation_states is not None else "charge_balanced"
    except Exception as first_error:
        fallback = {"Na": 1, "O": -2, "Fe": 3, "Mn": 4}
        try:
            ranked = _rank(disordered, full, x, limit, fallback, layer_groups, na_sites)
            scheme = "fallback_Na1_O-2_Fe3_Mn4"
        except Exception as fallback_error:
            raise RuntimeError(
                f"charge-balanced/provided attempt failed ({first_error}); "
                f"fixed-charge retry failed ({fallback_error})"
            ) from fallback_error
    return [(structure, energy, scheme, raw_rank) for structure, energy, raw_rank in ranked]


def _rank(disordered, full, x, limit, oxidation_states, layer_groups, na_sites):
    trial = disordered.copy()
    if oxidation_states is not None:
        trial.add_oxidation_state_by_element(oxidation_states)
    else:
        # Guessing may silently assign Na0+ to partial Na; use charge balance.
        _assign_charge_balanced_average_tm_valence(
            trial,
            Fraction(str(x)),
            ValueError("explicit charge balance"),
        )
    transformer = OrderDisorderedStructureTransformation(algo=2, no_oxi_states=False)
    requested = max(limit, 10)
    while True:
        ranked = transformer.apply_transformation(trial, return_ranked_list=requested)
        if isinstance(ranked, dict):
            ranked = [ranked]
        result = []
        seen_occupancies = set()
        for raw_rank, item in enumerate(ranked):
            energy = item.get("energy")
            if not isinstance(energy, (int, float)):
                raise RuntimeError("electrostatic ordering returned no Ewald score")
            structure = item["structure"].copy()
            if not structure.is_ordered:
                raise RuntimeError("electrostatic ordering returned a disordered structure")
            if _has_na_in_each_layer(structure, layer_groups, full, na_sites):
                occupancy_key = tuple(
                    sorted(
                        tuple(round(float(coord % 1.0), 8) for coord in site.frac_coords)
                        for site in structure
                        if site.is_ordered and site.specie.symbol == "Na"
                    )
                )
                if occupancy_key in seen_occupancies:
                    continue
                seen_occupancies.add(occupancy_key)
                structure.remove_oxidation_states()
                result.append((structure, float(energy), raw_rank))
        if len(result) >= limit or len(ranked) < requested or requested >= 10000:
            break
        requested = min(requested * 5, 10000)
    if not result:
        raise RuntimeError("静电能排序候选中没有满足每个 Na 层至少一个 Na 的构型；branch 已跳过")
    return result[:limit]


def _na_layer_groups(full, na_sites, z_tolerance=0.02):
    """Group candidate Na sites into periodic layers along c."""
    import numpy as np

    indexed = sorted(
        (float(full[index].frac_coords[2] % 1.0), position)
        for position, index in enumerate(na_sites)
    )
    groups = []
    for z, position in indexed:
        if not groups or z - groups[-1][-1][0] > z_tolerance:
            groups.append([(z, position)])
        else:
            groups[-1].append((z, position))
    if len(groups) > 1 and (groups[0][0][0] + 1.0 - groups[-1][-1][0]) <= z_tolerance:
        groups[0] = groups[-1] + groups[0]
        groups.pop()
    if not groups:
        raise ValueError("母结构中没有 Na 候选位点")
    return [tuple(position for _, position in group) for group in groups]


def _has_na_in_each_layer(ordered, layer_groups, full, na_sites):
    na_z = [
        float(site.frac_coords[2] % 1.0)
        for site in ordered
        if site.is_ordered and site.specie.symbol == "Na"
    ]
    for group in layer_groups:
        center = _layer_center(group, full, na_sites)
        if not any(abs(((z - center) + 0.5) % 1.0 - 0.5) <= 0.02 for z in na_z):
            return False
    return len(na_z) >= len(layer_groups)


def _layer_center(group, full, na_sites):
    import numpy as np

    zs = np.asarray([float(full[na_sites[position]].frac_coords[2] % 1.0) for position in group])
    angles = zs * (2 * np.pi)
    return float(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean()) / (2 * np.pi) % 1.0)

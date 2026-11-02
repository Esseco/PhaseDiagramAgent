"""Derive adjacent-Na structures from a verified near-hull parent frame."""

import hashlib
from fractions import Fraction
from pathlib import Path
import random

import numpy as np
from scipy.optimize import linear_sum_assignment

from phase_agent.science.structures.boundary_utils import load_structure, normalize_fraction
from phase_agent.science.structures.identify_branch import calculate_x, extract_T


def verify_parent_file(source):
    path = Path(source["structure_path"])
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != str(source["structure_sha256"]).removeprefix("sha256:"):
        raise ValueError("near_hull_parent_structure_checksum_mismatch")
    return path


def verify_parent_matches_branch(source, branch, boundary, references):
    parent = load_structure(verify_parent_file(source))
    actual_T, _ = extract_T(parent, boundary["TM_ratio"])
    oxygen = parent.composition.get_el_amt_dict().get("O", 0)
    expected_oxygen = (branch.get("composition") or {}).get("O")
    if (
        source["phase"] != branch["P"]
        or source["branch_H"] != branch["H"]
        or list(actual_T) != list(branch["T"])
        or Fraction(calculate_x(parent)) != Fraction(source["branch_x"])
        or (expected_oxygen is not None and abs(oxygen - expected_oxygen) > 1e-8)
    ):
        raise ValueError("near_hull_parent_structure_does_not_match_recorded_branch")
    return parent


def inherited_na_structures(branch, full_reference, boundary, references, limit, seed):
    source = branch.get("near_hull_parent_evidence")
    if not source:
        raise ValueError("composition_requires_near_hull_structure_evidence")
    parent = verify_parent_matches_branch(source, branch, boundary, references)
    reference_sites = [
        site.frac_coords
        for site in full_reference
        if site.is_ordered and site.specie.symbol == "Na"
    ]
    parent_sites = [
        site.frac_coords for site in parent if site.is_ordered and site.specie.symbol == "Na"
    ]
    if len(parent_sites) > len(reference_sites):
        raise ValueError("near_hull_parent_na_site_mapping_failed")
    occupied_reference = set()
    if parent_sites:
        distances = np.array(
            [
                [parent.lattice.get_distance_and_image(a, b)[0] for b in reference_sites]
                for a in parent_sites
            ]
        )
        left, right = linear_sum_assignment(distances)
        if len(left) != len(parent_sites) or np.any(distances[left, right] > 1.2):
            raise ValueError("near_hull_parent_na_site_mapping_failed")
        occupied_reference = set(int(index) for index in right)
    full = parent.copy()
    for index, coords in enumerate(reference_sites):
        if index not in occupied_reference:
            full.append("Na", coords, coords_are_cartesian=False)
    na_indices = [
        i for i, site in enumerate(full) if site.is_ordered and site.specie.symbol == "Na"
    ]
    occupied_count = len(parent_sites)
    target = Fraction(normalize_fraction(branch["x"]))
    oxygen_count = sum(site.is_ordered and site.specie.symbol == "O" for site in full)
    target_count = target * oxygen_count / 2
    if target_count.denominator != 1:
        raise ValueError("near_hull_neighbor_na_count_not_integer")
    delta = target_count.numerator - occupied_count
    if abs(delta) != 1:
        raise ValueError("near_hull_neighbor_must_add_or_remove_one_na")
    eligible = range(occupied_count) if delta < 0 else range(occupied_count, len(na_indices))
    choices = list(eligible)
    random.Random(seed).shuffle(choices)
    output = []
    for position in choices[:limit]:
        occupied = set(range(occupied_count))
        if delta < 0:
            occupied.remove(position)
        else:
            occupied.add(position)
        structure = full.copy()
        structure.remove_sites(
            [index for pos, index in enumerate(na_indices) if pos not in occupied]
        )
        vector = [int(pos in occupied) for pos in range(len(na_indices))]
        output.append((structure, full.copy(), vector))
    return output

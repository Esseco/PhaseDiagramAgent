"""Identify layered O/P stacking from the six nearest oxygen neighbours."""

from __future__ import annotations

from collections import Counter

import numpy as np

from phase_agent.science.structures.identify_branch import _cluster_layers


def identify_layered_phase_fast(structure, *, cation="Na", layer_tolerance=0.08):
    """Return O3/P3/OP2-style labels using O:6 versus T:6 Na sites.

    The relative orientation of the oxygen triangles above and below a mobile
    ion distinguishes an octahedron (staggered) from a trigonal prism (aligned).
    Ambiguous local environments return X so the caller can use the existing
    coordination-environment classifier.
    """
    indices = [
        i for i, site in enumerate(structure) if site.is_ordered and site.specie.symbol == cation
    ]
    if not indices:
        raise ValueError(f"结构中没有 {cation}，需用母结构识别相")
    if float(structure.composition.get("O", 0)) < 2:
        return {
            "phase": "X",
            "method": "oxygen_triangle_O6_T6",
            "reason": "insufficient_oxygen_for_layered_oxide",
        }
    a, b = np.asarray(structure.lattice.matrix[:2], dtype=float)
    normal = np.cross(a, b)
    normal /= np.linalg.norm(normal)
    x_axis = a - np.dot(a, normal) * normal
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(normal, x_axis)
    labels = _cluster_layers([float(structure[i].frac_coords[2]) for i in indices], layer_tolerance)

    site_types = {}
    for index in indices:
        site = structure[index]
        oxygen = sorted(
            (
                neighbor
                for neighbor in structure.get_neighbors(site, 4.0)
                if neighbor.specie.symbol == "O"
            ),
            key=lambda neighbor: neighbor.nn_distance,
        )[:6]
        if len(oxygen) != 6:
            site_types[index] = "X"
            continue
        vectors = [np.asarray(neighbor.coords - site.coords, dtype=float) for neighbor in oxygen]
        top = [vector for vector in vectors if np.dot(vector, normal) > 0]
        bottom = [vector for vector in vectors if np.dot(vector, normal) < 0]
        if len(top) != 3 or len(bottom) != 3:
            site_types[index] = "X"
            continue

        def triangle_order(vectors):
            angles = [
                np.arctan2(np.dot(vector, y_axis), np.dot(vector, x_axis)) for vector in vectors
            ]
            return np.mean(np.exp(3j * np.asarray(angles)))

        top_order, bottom_order = triangle_order(top), triangle_order(bottom)
        magnitude = abs(top_order) * abs(bottom_order)
        correlation = (
            (top_order * np.conj(bottom_order)).real / magnitude if magnitude > 0.1 else 0.0
        )
        site_types[index] = "P" if correlation > 0.25 else "O" if correlation < -0.25 else "X"

    layers = []
    for label in range(max(labels) + 1):
        counts = Counter(
            site_types[index]
            for index, current in zip(indices, labels, strict=True)
            if current == label and site_types[index] != "X"
        )
        if not counts:
            layers.append("X")
            continue
        ordered = counts.most_common()
        layers.append(ordered[0][0] if len(ordered) == 1 or ordered[0][1] > ordered[1][1] else "X")
    types = set(layers)
    phase = (
        f"O{len(layers)}"
        if types == {"O"}
        else f"P{len(layers)}"
        if types == {"P"}
        else f"OP{len(layers)}"
        if types == {"O", "P"}
        else "X"
    )
    return {
        "phase": phase,
        "method": "oxygen_triangle_O6_T6",
        "cation_layer_types": layers,
        "ambiguous_sites": sum(kind == "X" for kind in site_types.values()),
    }

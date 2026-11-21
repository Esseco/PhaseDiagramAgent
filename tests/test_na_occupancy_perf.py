import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from phase_agent.science.structures.initialize_branch_structures import _na_occupancy_indices


def old_indices(full, ordered, na_indices):
    positions = [s.frac_coords for s in ordered if s.is_ordered and s.specie.symbol == 'Na']
    chosen = [i for i, idx in enumerate(na_indices) if any(
        np.allclose((p-full[idx].frac_coords)-np.rint(p-full[idx].frac_coords), 0, atol=1e-5)
        for p in positions
    )]
    if len(chosen) != len(positions):
        raise RuntimeError('unmapped')
    return tuple(chosen)


def test_periodic_mapping_matches_previous_algorithm():
    rng = np.random.default_rng(34)
    points = rng.uniform(0, 1, size=(48, 3))
    full = Structure(Lattice.cubic(10), ['Na']*48, points)
    for size in (0, 1, 12, 32, 48):
        chosen = rng.choice(48, size=size, replace=False)
        positions = points[chosen] + rng.integers(-1, 2, size=(size, 3))
        ordered = Structure(Lattice.cubic(10), ['Na']*size, positions)
        assert _na_occupancy_indices(full, ordered, list(range(48))) == old_indices(full, ordered, list(range(48)))


def test_ambiguous_and_missing_sites_still_fail():
    full = Structure(Lattice.cubic(10), ['Na', 'Na'], [[0,0,0],[0.25,0.25,0.25]])
    duplicate = Structure(Lattice.cubic(10), ['Na','Na'], [[0,0,0],[1,0,0]])
    missing = Structure(Lattice.cubic(10), ['Na'], [[0.5,0.5,0.5]])
    for ordered in (duplicate, missing):
        with pytest.raises(RuntimeError, match='Na'):
            _na_occupancy_indices(full, ordered, [0,1])

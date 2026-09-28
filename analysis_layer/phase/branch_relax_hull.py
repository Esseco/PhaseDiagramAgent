"""Versioned relaxed-energy pool and convex-envelope queries in eV/atom."""

import hashlib
import json
import math
import statistics
from copy import deepcopy

from pymatgen.core import Composition


ENERGY_BASIS = 'total_eV_with_actual_cell_composition'


def build_relax_hull(records, *, model_version, system_id=None):
    rows = [deepcopy(r) for r in records if r.get('model_version') == model_version
            and r.get('converged') is True and r.get('energy_unit') == 'eV'
            and r.get('energy') is not None and math.isfinite(float(r['energy']))]
    version = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()[:16]
    return {'version': version, 'system_id': system_id, 'model_version': model_version,
            'records': rows, 'energy_basis': ENERGY_BASIS}


def hull_energy_per_atom(pool, composition):
    """Query any composition inside the sampled composition domain.

    Linear programming also supports constrained chemical systems without pure
    elemental endpoints. Outside the domain is unknown, never extrapolated.
    """
    import numpy as np
    from scipy.optimize import linprog
    target = Composition(composition).fractional_composition
    elements = sorted({str(e) for r in pool['records'] for e in Composition(r['composition'])}
                      | {str(e) for e in target})
    if not pool['records']:
        return None
    compositions = [Composition(r['composition']) for r in pool['records']]
    costs = [float(r['energy']) / c.num_atoms for r, c in zip(pool['records'], compositions)]
    matrix = [[c.fractional_composition[e] for c in compositions] for e in elements]
    result = linprog(costs, A_eq=np.array(matrix + [[1.] * len(costs)]),
                     b_eq=[target[e] for e in elements] + [1.], bounds=(0, None), method='highs')
    return float(result.fun) if result.success else None


def rank_relaxed_branches(candidates, pool, *, uncertainty_weight=1.0):
    ranked, missing = [], []
    for branch in candidates:
        rows = [r for r in pool['records'] if r['branch_id'] == branch['branch_id']]
        if not rows:
            missing.append(branch['branch_id']); continue
        best = min(rows, key=lambda r: float(r['energy']) / Composition(r['composition']).num_atoms)
        energy = float(best['energy']) / Composition(best['composition']).num_atoms
        reference = hull_energy_per_atom(pool, best['composition'])
        relaxed_energies = [float(row['energy']) / Composition(row['composition']).num_atoms
                            for row in rows]
        uncertainty = statistics.pstdev(relaxed_energies) if len(relaxed_energies) >= 3 else None
        gap = energy - reference if reference is not None else None
        allocation = (gap - uncertainty_weight * float(uncertainty)
                      if gap is not None and uncertainty is not None else gap)
        ranked.append({**branch, 'structure_id': best['structure_id'],
                       'structure_path': best['structure_path'],
                       'atom_count': Composition(best['composition']).num_atoms,
                       'relaxed_energy_per_atom': energy, 'hull_reference_energy_per_atom': reference,
                       'relaxed_ehull': gap,
                       'branch_relax_sample_count': len(relaxed_energies),
                       'branch_energy_std_per_atom': (float(uncertainty) if uncertainty is not None else None),
                       'relax_evidence_status': ('unknown' if gap is None else
                           'complete' if uncertainty is not None else 'energy_only'),
                       'allocation_score': allocation})
    return ranked, missing

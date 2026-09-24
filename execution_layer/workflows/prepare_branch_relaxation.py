"""Prepare version-specific, budgeted Relax screening before MC allocation."""

from copy import deepcopy
import hashlib
import json
import random

from scientific_layer.structures.initialize_branch_structures import initialize_branch_structures
from data_layer.ledger.register_candidate_batch import register_candidate_batch
from execution_layer.budget.reserve_budget import reserve_budget
from analysis_layer.phase.branch_relax_hull import ENERGY_BASIS, build_relax_hull, rank_relaxed_branches
from data_layer.ledger.branch_energy_pool_ledger import load_branch_energy_pools, save_branch_energy_pool


def prepare_branch_relaxation(candidates, state, context):
    from execution_layer.budget.estimate_stage_cost import estimate_stage_cost
    config = context['effective_config']; manager = context['manager']
    settings = config['bohb']; count = min(3, int(settings.get('relax_structures_per_branch', 3)))
    if count <= 0:
        raise ValueError('relax_structures_per_branch must be positive and is capped at 3')
    mlip = config.get('mlip') or {}
    version = mlip.get('version') or mlip.get('name')
    if not version and mlip.get('model_path'):
        from pathlib import Path
        version = Path(mlip['model_path']).stem
    if not version:
        raise ValueError('confirmed MLIP version required for Relax screening')
    current = deepcopy(state); pending = []; unavailable = []
    system_id = (config.get('system_config') or {}).get('system_id')
    ledger_path = config.get('branch_energy_pool_ledger_path')
    if ledger_path:
        restored = load_branch_energy_pools(ledger_path, system_id=system_id,
                                            mlip_version=version, energy_basis=ENERGY_BASIS)
        for saved_pool in restored:
            current.setdefault('branch_hull_batches', {}).setdefault(saved_pool['version'], saved_pool)
    records = []
    for branch in candidates:
        if len(branch.get('structure_ids') or []) < count:
            initialized = initialize_branch_structures([branch], manager.boundary, context['phase_references'],
                initial_states_per_branch=count, seed=config.get('seed', 42))
            register_candidate_batch(manager, initialized, structure_directory=config['structure_directory'],
                                     ledger_path=config.get('ledger_path'))
        ids = _screen_structure_ids(manager, branch['branch_id'], count=count,
                                    seed=int(config.get('seed', 42)))
        relax_settings = deepcopy((config.get('mlip') or {}).get('relax_parameters') or {})
        settings_id = hashlib.sha256(json.dumps(relax_settings, sort_keys=True,
                                                default=str).encode()).hexdigest()[:12]
        for sid in ids:
            key = f'relax-screen:{version}:{settings_id}:{sid}'
            matches = [t for t in current.get('tasks', []) if t.get('task_key') == key]
            if matches:
                task = matches[-1]; out = task.get('outputs') or {}
                normal_stop = out.get('relax_stopped_normally', task.get('converged') is True)
                if task.get('status') == 'completed' and normal_stop:
                    composition = out.get('composition')
                    if composition and out.get('structure_path'):
                        records.append({'branch_id': branch['branch_id'], 'structure_id': sid,
                            'structure_path': out['structure_path'], 'composition': composition,
                            'energy': out.get('energy'), 'energy_unit': out.get('energy_unit'),
                            'relax_stopped_normally': True, 'converged': True, 'model_version': version})
                    else:
                        unavailable.append(sid)
                elif task.get('status') in {'pending', 'running'}:
                    pending.append(task)
                else:
                    unavailable.append(sid)
                continue
            record = manager.data['structures'][sid]
            from pymatgen.core import Structure
            atoms = len(Structure.from_file(record['source_path']))
            cost = estimate_stage_cost('relax_and_feature', atom_count=atoms, budgets=config['budgets'])['value']
            reservation = reserve_budget(current, task_key=key, stage='relax_and_feature', amount=cost,
                limits=config['budgets'], config_version=context['config_version'], model_version=version)
            if reservation['status'] != 'reserved':
                unavailable.append(sid); continue
            current = reservation['state']
            task = {'task_id': 'RELAX-' + hashlib.sha256(key.encode()).hexdigest()[:12],
                    'task_key': key, 'structure_id': sid, 'object_id': sid,
                    'branch_id': branch['branch_id'], 'stage': 'relax_and_feature', 'status': 'pending',
                    'model_version': version, 'config_version': context['config_version'],
                    'planned_relative_cost': cost, 'parameters': relax_settings,
                    'screening_basis': 'legal_electrostatic_top10_fixed_seed'}
            current.setdefault('tasks', []).append(task)
            current.setdefault('pending_tasks', []).append(task); pending.append(task)
    if pending or unavailable:
        return {'state': current, 'status': 'screening_pending' if pending else 'screening_incomplete',
                'candidates': [], 'unavailable': unavailable}
    pool = build_relax_hull(records, model_version=version, system_id=system_id)
    ranked, missing = rank_relaxed_branches(candidates, pool,
        uncertainty_weight=float(settings.get('uncertainty_weight', 1.0)))
    current.setdefault('branch_hull_batches', {})[pool['version']] = pool
    current['current_branch_hull_version'] = pool['version']
    if ledger_path:
        save_branch_energy_pool(pool, ledger_path)
        current['branch_energy_pool_ledger_path'] = str(ledger_path)
    return {'state': current, 'status': 'ready' if not missing else 'uncertainty_unavailable',
            'candidates': ranked, 'pool': pool, 'unavailable': missing}


def _screen_structure_ids(manager, branch_id, *, count, seed):
    """Sample without replacement from the ten lowest known electrostatic energies."""
    rows = []
    for structure_id in manager.data['branches'][branch_id].get('structure_ids') or []:
        record = manager.data['structures'][structure_id]
        metadata = record.get('metadata') or {}
        if metadata.get('legal') is False or metadata.get('legality') in {'illegal', 'rejected'}:
            continue
        energy = metadata.get('electrostatic_energy')
        known = isinstance(energy, (int, float))
        rows.append((0 if known else 1, float(energy) if known else 0.0, structure_id))
    top = [row[2] for row in sorted(rows)[:10]]
    if len(top) <= count:
        return top
    digest = int(hashlib.sha256(branch_id.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed + digest)
    return sorted(rng.sample(top, count))

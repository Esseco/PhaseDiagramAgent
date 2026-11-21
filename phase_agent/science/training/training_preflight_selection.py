"""Staging for a fail-closed, user-scoped training selection policy."""

from copy import deepcopy
from collections import defaultdict
import hashlib
import math
import random


def structure_conflicts(training, test):
    """Conservatively reject symmetry/supercell matches across the split."""
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.analysis.structure_matcher import StructureMatcher

    def structure(row):
        value = row.get('structure')
        if isinstance(value, dict):
            return Structure.from_dict(value)
        if isinstance(value, Structure):
            return value
        return AseAtomsAdaptor.get_structure(value)

    prepared = [(r, structure(r)) for r in training]
    matcher = StructureMatcher(primitive_cell=True, scale=False, attempt_supercell=True)
    conflicts = []
    for row in test:
        left = structure(row)
        for other, right in prepared:
            if left.composition.reduced_composition != right.composition.reduced_composition:
                continue
            if matcher.fit(left, right):
                conflicts.append((row['data_id'], other['data_id']))
    return conflicts


def group_equivalent_training_records(records, settings):
    """Keep matching structures and all their source groups in the same fold."""
    from phase_agent.science.training.prepare_mace_finetune import _group_key
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    grouped = defaultdict(list)
    for row in records:
        group = _group_key(row, settings)
        value = row.get('structure')
        if isinstance(value, dict):
            value = Structure.from_dict(value)
        elif not isinstance(value, Structure):
            value = AseAtomsAdaptor.get_structure(value)
        grouped[str(value.composition.reduced_composition)].append((group, value))
    parents = {group: group for values in grouped.values() for group, _ in values}
    def find(group):
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group
    def merge(a, b):
        a, b = find(a), find(b)
        if a != b:
            parents[max(a, b)] = min(a, b)
    matcher = StructureMatcher(primitive_cell=True, scale=False, attempt_supercell=True)
    matches = 0
    for values in grouped.values():
        for i, (ga, sa) in enumerate(values):
            for gb, sb in values[i + 1:]:
                if find(ga) != find(gb) and matcher.fit(sa, sb):
                    merge(ga, gb)
                    matches += 1
    components = defaultdict(list)
    for group in parents:
        components[find(group)].append(group)
    mapping = {}
    for groups in components.values():
        identifier = hashlib.sha256('\n'.join(sorted(groups)).encode()).hexdigest()[:20]
        mapping.update({group: identifier for group in groups})
    if len(components) < 5:
        raise ValueError('independent_cv_insufficient_structure_groups')
    output = []
    for row in records:
        copy = deepcopy(row)
        copy['independent_group_id'] = mapping[_group_key(row, settings)]
        output.append(copy)
    return output, {'source_group_count': len(parents), 'independent_group_count': len(components),
                    'matched_group_merges': matches, 'source_to_independent_group': mapping}



def validate_exception_frame(task, record):
    """Bind an authorized static DFT label to its exact available input frame."""
    from pathlib import Path
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure

    if task.get('stage') != 'dft_single_point' or record.get('stage') != 'dft_single_point':
        raise ValueError('training_exception_requires_static_dft')
    path = Path(task.get('structure_path') or '')
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != task.get('input_structure_sha256'):
        raise ValueError('training_exception_input_frame_unavailable_or_changed')
    incoming = Structure.from_file(path)
    value = record.get('structure')
    actual = Structure.from_dict(value) if isinstance(value, dict) else value
    matcher = StructureMatcher(ltol=1e-4, stol=1e-4, angle_tol=1e-3,
                               primitive_cell=False, scale=False)
    if not isinstance(actual, Structure) or not matcher.fit(incoming, actual):
        raise ValueError('training_exception_dft_frame_mismatch')


def selected_records(state, policy):
    from phase_agent.science.training.cumulative_training_records import cumulative_training_records
    from phase_agent.science.training.prepare_mace_finetune import _to_atoms
    from phase_agent.evidence.dft_extrapolation import failed_dft_task_ids

    rows = cumulative_training_records(state)
    identities = {r.get('data_id') for r in rows}
    failed_frames = failed_dft_task_ids(state)
    specs = policy.get('include_extrapolated_dft') or []
    if not isinstance(specs, list):
        raise ValueError('training_exception_invalid')
    tasks = {r.get('task_id'): r for r in state.get('tasks') or []}
    history = [*(state.get('dft_training_records') or []), *(state.get('new_dft_records') or [])]
    for spec in specs:
        if not isinstance(spec, dict) or not spec.get('user_instruction'):
            raise ValueError('training_exception_missing_user_authorization')
        task_id = spec.get('task_id')
        task = tasks.get(task_id) or {}
        model = task.get('extrapolation_model_version') or task.get('model_version')
        if (task_id not in failed_frames or task.get('status') != 'completed'
            or task.get('input_structure_sha256') != spec.get('input_structure_sha256')
            or model != spec.get('model_version')):
            raise ValueError('training_exception_identity_or_status_mismatch')
        candidates = [x for x in history if x.get('task_id') == task_id]
        if not candidates or any(candidate != candidates[0] for candidate in candidates[1:]):
            raise ValueError('training_exception_missing_or_ambiguous_record')
        record = candidates[0]
        if record.get('checks_passed') is not True or record.get('training_ready') is not True:
            raise ValueError('training_exception_dft_not_validated')
        _to_atoms(record, {})
        validate_exception_frame(task, record)
        if record.get('data_id') not in identities:
            row = deepcopy(record)
            row['is_current_round'] = any(x.get('data_id') == row.get('data_id') for x in state.get('new_dft_records') or [])
            rows.append(row)
            identities.add(record.get('data_id'))
    return rows


def split_independent_holdout(records, base, settings, policy):
    from phase_agent.analysis.convergence.independent_evidence import geometry_key
    from phase_agent.science.training.prepare_mace_finetune import _group_key

    if base.get('training_provenance_complete') is not True:
        raise ValueError('independent_holdout_prior_provenance_incomplete')
    prior_ids = set(base.get('training_data_ids') or [])
    prior_geometry = set(base.get('training_structure_hashes') or [])
    if prior_ids - {row.get('data_id') for row in records}:
        raise ValueError('independent_holdout_prior_records_missing')
    groups = defaultdict(list)
    for row in records:
        if not row.get('data_id') or not geometry_key(row):
            raise ValueError('independent_holdout_identity_missing')
        groups[_group_key(row, settings)].append(row)
    protected = {x['task_id'] for x in policy.get('include_extrapolated_dft') or []}
    eligible = [key for key, values in groups.items()
                if all(r['data_id'] not in prior_ids and geometry_key(r) not in prior_geometry
                       and r.get('task_id') not in protected for r in values)]
    fraction = policy.get('independent_test_fraction', 0.1)
    if type(fraction) not in (int, float) or not 0 < fraction < 0.5:
        raise ValueError('independent_holdout_fraction_invalid')
    if len(eligible) < 2 or len(groups) - 1 < 5:
        raise ValueError('independent_holdout_insufficient_groups')
    random.Random(int(policy.get('seed', 2026))).shuffle(eligible)
    target = max(1, math.ceil(len(records) * fraction))
    rejected = set()
    while True:
        chosen, count = [], 0
        for key in eligible:
            if key in rejected:
                continue
            if len(groups) - len(chosen) <= 5:
                break
            chosen.append(key)
            count += len(groups[key])
            if count >= target:
                break
        if count < target:
            raise ValueError('independent_holdout_insufficient_unseen_records')
        held = set(chosen)
        training = [r for key, values in groups.items() if key not in held for r in values]
        test = [r for key, values in groups.items() if key in held for r in values]
        conflicts = structure_conflicts(training, test)
        conflict_ids = {left for left, _ in conflicts}
        if not conflict_ids:
            if {geometry_key(r) for r in training} & {geometry_key(r) for r in test}:
                raise ValueError('independent_holdout_geometry_overlap')
            return training, test, {'groups': sorted(held), 'data_ids': sorted(r['data_id'] for r in test),
                                    'fraction_requested': fraction, 'seed': int(policy.get('seed', 2026)),
                                    'rejected_overlap_groups': sorted(rejected)}
        blocked = {_group_key(r, settings) for r in test if r['data_id'] in conflict_ids}
        if not blocked - rejected:
            raise ValueError('independent_holdout_overlap_unresolvable')
        rejected.update(blocked)

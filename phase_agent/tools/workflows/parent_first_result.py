"""Validate parent-first HPC sampling results before any DFT registration."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_parent_first_result(input_dir, result_dir):
    """Check exact input binding, parent evidence, selected hashes and TM ordering.

    This is read-only and does not import data into a scientific ledger.
    """
    from ase.io import read
    from phase_agent.science.structures.tm_order import tm_order_preserved

    root, output = Path(input_dir).resolve(), Path(result_dir).resolve()
    manifest_path = root / 'input_manifest.json'
    input_data = json.loads(manifest_path.read_text(encoding='utf8'))
    result = json.loads((output / 'result.json').read_text(encoding='utf8'))
    expected_binding = {k: input_data[k] for k in ('scope_id', 'round_evidence_digest', 'baseline_hull_version')}
    expected_binding['input_manifest_sha256'] = _sha(manifest_path)
    if (input_data.get('schema') != 'parent-first-hpc-input-v1'
            or result.get('schema') != 'parent-first-perturbation-direct-v1'
            or result.get('binding') != expected_binding
            or result.get('model_sha256') != input_data.get('model_sha256')
            or (result.get('feature_model') or {}).get('model_sha256') != input_data.get('feature_model_sha256')):
        raise ValueError('parent_first_return_binding_invalid')
    parents = input_data.get('parents') or []
    by_id = {p.get('record_id'):p for p in parents}
    relaxed = result.get('parent_relaxations') or []
    if not parents or len(by_id) != len(parents) or len(relaxed) != len(parents):
        raise ValueError('parent_first_return_parent_inventory_invalid')
    checked = {}
    for row in relaxed:
        parent = by_id.get(row.get('parent_record_id'))
        monitor = row.get('monitor') or {}
        score = monitor.get('final_feature_difference')
        if (not parent or row['parent_record_id'] in checked or row.get('input_sha256') != parent.get('sha256')
                or not re.fullmatch(r'[a-f0-9]{64}', str(row.get('relaxed_sha256') or ''))
                or row.get('optimizer_converged') is not True
                or type(row.get('n_perturbations')) is not int
                or not 1 <= row['n_perturbations'] <= input_data['frames_per_parent']
                or monitor.get('extrapolation_failure') is not False or monitor.get('failures') != []
                or isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError('parent_first_return_relaxation_invalid')
        source = (root / parent['path']).resolve()
        if not source.is_relative_to(root) or not source.is_file() or _sha(source) != parent.get('sha256'):
            raise ValueError('parent_first_return_parent_source_invalid')
        checked[row['parent_record_id']] = (parent, row, source)
    if result.get('candidate_count') != sum(row['n_perturbations'] for row in relaxed):
        raise ValueError('parent_first_return_candidate_count_invalid')
    expected_refs = [(r.get('task_id'), r.get('sha256')) for r in input_data.get('references') or []]
    returned_refs = [(r.get('task_id'), r.get('sha256')) for r in result.get('references') or []]
    if returned_refs != expected_refs:
        raise ValueError('parent_first_return_references_invalid')
    for ref in input_data.get('references') or []:
        source = (root / ref['path']).resolve()
        if not source.is_relative_to(root) or not source.is_file() or _sha(source) != ref.get('sha256'):
            raise ValueError('parent_first_return_reference_source_invalid')
    selected = result.get('selected') or []
    target = input_data.get('target')
    if (type(target) is not int or not 1 <= target <= 100 or result.get('target') != target
            or len(selected) != target or (result.get('direct_report') or {}).get('n_samples') != target):
        raise ValueError('parent_first_return_selection_count_invalid')
    expected_names = {'result.json'}
    seen_hashes = set()
    rows = []
    for row in selected:
        name = row.get('path') or ''
        if not re.fullmatch(r'selected_[0-9]{4}\.vasp', name) or name in expected_names:
            raise ValueError('parent_first_return_selected_path_invalid')
        expected_names.add(name)
        parent, relaxation, source = checked.get(row.get('parent_record_id'), (None, None, None))
        target_path = output / name
        if (not parent or row.get('parent_sha256') != parent['sha256']
                or row.get('relaxed_sha256') != relaxation['relaxed_sha256']
                or any(row.get(key) != parent.get(key) for key in ('phase','branch_id','H','x_Na_per_O2'))
                or not target_path.resolve().is_relative_to(output)
                or not target_path.is_file() or _sha(target_path) != row.get('sha256')
                or row.get('sha256') in seen_hashes):
            raise ValueError('parent_first_return_selected_binding_invalid')
        original, atoms = read(source), read(target_path)
        if (sorted(original.get_chemical_symbols()) != sorted(atoms.get_chemical_symbols())
                or not tm_order_preserved(original, atoms)):
            raise ValueError('parent_first_return_tm_or_composition_invalid')
        seen_hashes.add(row['sha256'])
        rows.append({**row, 'structure_path':str(target_path), 'structure_sha256':row['sha256']})
    actual = {p.name for p in output.iterdir() if p.is_file()}
    if actual != expected_names or any(not p.is_file() for p in output.iterdir()):
        raise ValueError('parent_first_return_unexpected_files')
    return {'status':'verified', 'target':target, 'parent_count':len(parents),
            'reference_count':len(expected_refs), 'selected':rows,
            'result_sha256':_sha(output / 'result.json'), 'binding':expected_binding}

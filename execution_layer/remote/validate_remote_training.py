"""Portable independent GPU evaluation; run only through the scheduler."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def geometry(atoms):
    import numpy as np
    return (tuple(atoms.numbers), tuple(np.round(atoms.positions, 7).flat),
            tuple(np.round(atoms.cell.array, 7).flat), tuple(atoms.pbc))


def evaluate(plan):
    import numpy as np
    from ase.io import read
    from mace.calculators import MACECalculator
    root = Path(__file__).resolve().parent
    data = root / 'validation.xyz'
    if sha256(data) != plan['data_sha256']:
        raise ValueError('Validation data hash mismatch')
    frames = read(data, ':')
    if sha256(root / '_shared_data/train.xyz') != plan['training_sha256']:
        raise ValueError('Training data hash mismatch')
    training = read(root / '_shared_data/train.xyz', ':')
    train_geometries = {geometry(a) for a in training}
    if not frames or any(geometry(a) in train_geometries for a in frames):
        raise ValueError('Empty validation set or training/validation overlap')
    pairs = plan['ranking_pairs']
    for i, j in pairs:
        if i == j or not 0 <= i < len(frames) or not 0 <= j < len(frames):
            raise ValueError('Invalid ranking pair')
        if sorted(frames[i].get_chemical_symbols()) != sorted(frames[j].get_chemical_symbols()):
            raise ValueError('Ranking pairs must have the same composition and atom count')
    energy_key, force_key = plan['energy_key'], plan['forces_key']
    reference_e = [float(a.info[energy_key]) / len(a) for a in frames]
    reference_f = [np.asarray(a.arrays[force_key], dtype=float) for a in frames]
    if not np.isfinite(reference_e).all() or any(f.shape != (len(a), 3) or not np.isfinite(f).all()
            for a, f in zip(frames, reference_f)):
        raise ValueError('Invalid DFT labels')
    reports = {}
    for label in ('old_model', 'new_model'):
        model = plan[label]
        if sha256(model['model_path']) != model['sha256']:
            raise ValueError('Model hash mismatch: ' + label)
        options = {'head': model['mace_head']} if model.get('mace_head') else {}
        calc = MACECalculator(model_paths=model['model_path'], device='cuda', default_dtype='float64', **options)
        predicted_e, errors_f = [], []
        # Any failed structure fails the job; never silently drop difficult points.
        for atoms, forces in zip(frames, reference_f):
            copy = atoms.copy()
            copy.calc = calc
            predicted_e.append(float(copy.get_potential_energy()) / len(copy))
            errors_f.extend((copy.get_forces() - forces).ravel().tolist())
        error_e = np.asarray(predicted_e) - reference_e
        if not np.isfinite(error_e).all() or not np.isfinite(errors_f).all():
            raise ValueError('Nonfinite predictions')
        reversals = sum((predicted_e[i] - predicted_e[j]) * (reference_e[i] - reference_e[j]) < 0 for i, j in pairs)
        reports[label] = {'energy_mae': float(np.abs(error_e).mean()),
            'force_rmse': float(np.sqrt(np.square(errors_f).mean())),
            'critical_failure_fraction': 0.0, 'near_hull_ranking_reversals': int(reversals)}
    return {'request_id': plan['request_id'], 'data_sha256': plan['data_sha256'],
            'status': 'completed', 'structures': len(frames), 'metrics': reports}


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    plan = json.loads((root / 'validation_request.json').read_text())
    report = evaluate(plan)
    output = root.parent / 'results' / ('validation-' + plan['request_id'] + '.json')
    output.parent.mkdir(exist_ok=True)
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
    temporary.replace(output)

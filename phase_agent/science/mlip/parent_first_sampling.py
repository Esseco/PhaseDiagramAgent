"""HPC-side parent relaxation, Process_MLIP perturbations and DIRECT selection.

This worker does not submit jobs or write project ledgers. The caller provides
verified input files and is responsible for the project's transfer protocol.
"""
from __future__ import annotations

import hashlib
import json
import math
import tempfile
from pathlib import Path


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _feature_checkpoint_sha(path):
    """Match Process_MLIP's directory digest without loading TensorFlow."""
    path = Path(path)
    if not path.is_dir():
        raise ValueError('parent_first_feature_model_missing')
    files = sorted(p for p in path.rglob('*') if p.is_file())
    if not files:
        raise ValueError('parent_first_feature_model_missing')
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.relative_to(path).as_posix().encode() + b'\0')
        with file.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
    return digest.hexdigest()


def run_parent_first_sampling(parents, references, output_dir, *, model_path, model_sha256,
                              feature_model_path, feature_model_sha256=None, target=100,
                              frames_per_parent=36, seed=42, min_distance_A=1.0,
                              volume_ratio=(0.75, 1.25), checkpoint_dir=None, binding=None,
                              relax_fn=None, encoder_factory=None, extract_fn=None, sample_fn=None):
    """Relax parents, perturb valid final frames and export exactly `target` representatives.

    Inputs and model must be SHA256-bound. A failed or unverified parent aborts;
    scratch perturbations and features are not part of the exported result.
    Injectable functions allow small synthetic tests without a model or HPC.
    """
    from ase.io import read, write
    from Process_MLIP.nepkit_sampling import perturb_layered_oxide_structure
    from phase_agent.science.structures.tm_order import tm_order_preserved

    if (not parents or type(target) is not int or not 1 <= target <= 100
            or len({p.get('record_id') for p in parents}) != len(parents)
            or any(not p.get('record_id') for p in parents)):
        raise ValueError('parent_first_inputs_invalid')
    model_path = Path(model_path)
    if not model_path.is_file() or _sha(model_path) != model_sha256:
        raise ValueError('parent_first_model_sha256_mismatch')
    for row in [*parents, *references]:
        path = Path(row.get('path') or '')
        if not path.is_file() or _sha(path) != row.get('sha256'):
            raise ValueError('parent_first_input_sha256_mismatch')
    if len({r['sha256'] for r in parents}) != len(parents):
        raise ValueError('parent_first_duplicate_parent')
    # Validate the real DIRECT checkpoint before starting any expensive parent
    # relaxation. Injected encoders are used only by callers supplying a test or
    # alternate encoder and remain responsible for their own model metadata.
    if encoder_factory is None and feature_model_sha256:
        if _feature_checkpoint_sha(feature_model_path) != feature_model_sha256:
            raise ValueError('parent_first_feature_model_sha256_mismatch')
    if relax_fn is None:
        from phase_agent.science.mlip.mace_worker import run_mace_worker
        relax_fn = run_mace_worker
    if encoder_factory is None:
        from Process_MLIP.direct_sampling.features import M3GNetEmbedding
        encoder_factory = M3GNetEmbedding
    if extract_fn is None or sample_fn is None:
        from Process_MLIP.direct_sampling.workflow import extract, run
        extract_fn = extract_fn or extract
        sample_fn = sample_fn or run
    out = Path(output_dir)
    if out.exists():
        raise FileExistsError('parent_first_output_exists')
    out.parent.mkdir(parents=True, exist_ok=True)
    # Keep finished parent relaxations across interruptions; never overwrite an
    # incomplete or differently bound calculation.
    checkpoint = Path(checkpoint_dir) if checkpoint_dir is not None else out.with_name(out.name + '.parents')
    checkpoint_binding = {'schema': 'parent-first-checkpoint-v1', 'model_sha256': model_sha256,
               'parents': [{'record_id': p['record_id'], 'sha256': p['sha256']} for p in parents]}
    binding_path = checkpoint / 'binding.json'
    if checkpoint.exists():
        if not binding_path.is_file() or json.loads(binding_path.read_text(encoding='utf8')) != checkpoint_binding:
            raise ValueError('parent_first_checkpoint_binding_mismatch')
    else:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.mkdir()
        binding_path.write_text(json.dumps(checkpoint_binding, sort_keys=True), encoding='utf8')
    with tempfile.TemporaryDirectory(prefix='.parent-first-', dir=out.parent) as tmp:
        root = Path(tmp)
        work = root / 'work'
        work.mkdir()
        candidate_paths, identities, relaxations, relaxed_parents = [], {}, [], []
        for index, parent in enumerate(parents):
            src = Path(parent['path']).resolve()
            saved = checkpoint / ('parent_%04d' % index)
            result_path = saved / 'result.json'
            if saved.exists():
                if not result_path.is_file():
                    raise ValueError('parent_first_incomplete_relaxation_requires_review:' + parent['record_id'])
                result = json.loads(result_path.read_text(encoding='utf8'))
                if result.get('parent_sha256') != parent['sha256'] or result.get('model_sha256') != model_sha256:
                    raise ValueError('parent_first_relaxation_checkpoint_mismatch:' + parent['record_id'])
                result['structure_path'] = str(saved / 'relax' / 'initial_relaxed.vasp')
            else:
                saved.mkdir()
                result = relax_fn({'operation': 'relax', 'structure_path': str(src),
                    'output_directory': str(saved / 'relax'), 'model_path': str(model_path),
                    'model_version': 'mace-mh-1', 'parameters': {'device': 'cuda',
                    'mace_head': 'omat_pbe', 'mace_default_dtype': 'float64',
                    'optimizer': 'LBFGS', 'relax_cell': True, 'fmax': 0.05, 'relax_steps': 500}})
                result['parent_sha256'] = parent['sha256']
                result['model_sha256'] = model_sha256
            final = Path(result.get('structure_path') or '')
            if final.resolve() != (saved / 'relax' / 'initial_relaxed.vasp').resolve():
                raise ValueError('parent_first_relaxation_path_invalid:' + parent['record_id'])
            monitor = result.get('extrapolation_monitor') or {}
            score = monitor.get('final_feature_difference')
            if (result.get('status') != 'completed' or result.get('optimizer_converged') is not True
                    or monitor.get('extrapolation_failure') is not False or monitor.get('failures') != []
                    or isinstance(score, bool) or not isinstance(score, (float, int))
                    or not math.isfinite(score) or not 0 <= score <= 1
                    or not final.is_file() or result.get('structure_checksum') != _sha(final)):
                raise ValueError('parent_first_relaxation_unverified:' + parent['record_id'])
            if not tm_order_preserved(read(src), read(final)):
                raise ValueError('parent_first_tm_order_changed:' + parent['record_id'])
            if not result_path.exists():
                temporary_result = saved / 'result.json.part'
                temporary_result.write_text(json.dumps(result, ensure_ascii=False, default=str), encoding='utf8')
                temporary_result.replace(result_path)
            relaxed_parents.append((parent, final, monitor))
        # All parents finish relaxation before any perturbation is generated.
        for index, (parent, final, monitor) in enumerate(relaxed_parents):
            dest = work / ('parent_%04d' % index)
            dest.mkdir()
            generated = perturb_layered_oxide_structure(final, dest / 'perturb',
                frames_per_parent=frames_per_parent, seed=seed + index,
                min_distance_A=min_distance_A, volume_ratio=volume_ratio,
                metadata={'parent_record_id': parent['record_id'], 'parent_sha256': parent['sha256'],
                          'relaxed_sha256': _sha(final), 'branch_id': parent.get('branch_id'),
                          'phase': parent.get('phase'), 'H': parent.get('H'),
                          'x_Na_per_O2': parent.get('x_Na_per_O2')})
            path = str(Path(generated['structures']).resolve())
            manifest = json.loads(Path(generated['manifest']).read_text(encoding='utf8'))
            if len(manifest['identities']) != generated['n_frames']:
                raise ValueError('parent_first_perturbation_manifest_invalid')
            candidate_paths.append(path)
            identities[path] = manifest['identities']
            relaxations.append({'parent_record_id': parent['record_id'], 'input_sha256': parent['sha256'],
                                'relaxed_sha256': _sha(final), 'optimizer_converged': True,
                                'n_perturbations': generated['n_frames'], 'monitor': monitor})
        candidate_count = sum(len(v) for v in identities.values())
        if candidate_count < target:
            raise ValueError('parent_first_insufficient_candidates')
        reference_paths = [str(Path(x['path']).resolve()) for x in references]
        if any(len(read(path, index=':')) != 1 for path in reference_paths):
            raise ValueError('parent_first_reference_requires_single_frame')
        cache = work / 'features.h5'
        encoder = encoder_factory(feature_model_path)
        feature_model = getattr(encoder, 'metadata', {}) or {}
        if feature_model_sha256 and feature_model.get('model_sha256') != feature_model_sha256:
            raise ValueError('parent_first_feature_model_sha256_mismatch')
        # Compute each relaxed parent and its perturbations in the same feature
        # cache, then remove parents and out-of-range perturbations before DIRECT.
        import h5py
        import numpy as np
        parent_paths = [str(Path(final).resolve()) for _, final, _ in relaxed_parents]
        extract_fn(reference_paths + parent_paths + candidate_paths, encoder, cache)
        with h5py.File(cache, 'r+') as h:
            features = h['features'][:]
            indices = h['indices'][:]
            nref, nparent = len(reference_paths), len(parent_paths)
            parent_features = {}
            for source, frame in indices:
                if nref <= source < nref + nparent:
                    if frame != 0 or int(source) in parent_features:
                        raise ValueError('parent_first_parent_feature_invalid')
                    parent_features[int(source)] = None
            for position, (source, frame) in enumerate(indices):
                if nref <= source < nref + nparent:
                    parent_features[int(source)] = features[position]
            if len(parent_features) != nparent:
                raise ValueError('parent_first_parent_feature_missing')
            kept, differences = [], {}
            for position, (source, frame) in enumerate(indices):
                source, frame = int(source), int(frame)
                if source < nref:
                    kept.append(position)
                elif source >= nref + nparent:
                    parent_index = source - nref - nparent
                    if not 0 <= parent_index < nparent:
                        raise ValueError('parent_first_feature_source_invalid')
                    delta = float(np.linalg.norm(features[position] - parent_features[nref + parent_index]))
                    if not math.isfinite(delta):
                        raise ValueError('parent_first_perturbation_feature_invalid')
                    differences[(candidate_paths[parent_index], frame)] = delta
                    if delta <= 1.0:
                        kept.append(position)
            accepted = len(kept) - nref
            if accepted < target:
                raise ValueError('parent_first_insufficient_candidates_after_feature_filter')
            h['features'].resize((len(kept), features.shape[1]))
            h['features'][:] = features[kept]
            del h['indices']
            h.create_dataset('indices', data=indices[kept])
        selection_dir = work / 'direct'
        report = sample_fn(cache, selection_dir, target, reference_count=len(reference_paths))
        selected = json.loads((selection_dir / 'selected_indices.json').read_text(encoding='utf8'))
        if len(selected) != target or report.get('n_samples') != target:
            raise ValueError('parent_first_direct_count_not_met')
        exported = root / 'export'
        exported.mkdir()
        rows = []
        seen = set()
        for index, item in enumerate(selected):
            source = str(Path(item['source']).resolve())
            frame = item.get('frame')
            if source not in identities or type(frame) is not int or not 0 <= frame < len(identities[source]):
                raise ValueError('parent_first_selected_frame_invalid')
            key = (source, frame)
            if key in seen:
                raise ValueError('parent_first_selected_duplicate')
            seen.add(key)
            if differences.get(key, float('inf')) > 1.0:
                raise ValueError('parent_first_selected_feature_filter_invalid')
            path = exported / ('selected_%04d.vasp' % index)
            write(path, read(source, index=frame), format='vasp', direct=True)
            rows.append({'path': path.name, 'sha256': _sha(path), **identities[source][frame],
                         'perturbation_feature_difference': differences[key]})
        output = {'schema': 'parent-first-perturbation-direct-v1', 'target': target,
                  'binding': binding or {},
                  'model_sha256': model_sha256, 'feature_model': feature_model, 'references': references,
                  'parent_relaxations': relaxations, 'candidate_count': candidate_count,
                  'selected': rows, 'direct_report': report,
                  'perturbation_feature_filter': {'threshold': 1.0, 'accepted': accepted,
                      'rejected': candidate_count - accepted}}
        (exported / 'result.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf8')
        exported.rename(out)
    return {'status': 'completed', 'output_dir': str(out), 'selected_count': len(rows),
            'candidate_count': candidate_count, 'submitted': False}


def main(argv=None):
    """Run from a checksummed HPC input bundle; all input paths are relative."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--checkpoint-dir', required=True)
    args = parser.parse_args(argv)
    path = Path(args.input).resolve()
    data = json.loads(path.read_text(encoding='utf8'))
    if data.get('schema') != 'parent-first-hpc-input-v1':
        raise ValueError('parent_first_input_schema_invalid')
    def bind(row):
        source = (path.parent / row['path']).resolve()
        if not source.is_relative_to(path.parent) or not source.is_file():
            raise ValueError('parent_first_input_path_invalid')
        return {**row, 'path': str(source)}
    if not data.get('feature_model_sha256'):
        raise ValueError('parent_first_feature_model_sha256_required')
    output = Path(args.output).resolve()
    checkpoints = Path(args.checkpoint_dir).resolve()
    if checkpoints.is_relative_to(output.parent):
        raise ValueError('parent_first_checkpoint_inside_results')
    binding = {key: data[key] for key in ('scope_id', 'round_evidence_digest', 'baseline_hull_version')}
    binding['input_manifest_sha256'] = _sha(path)
    result = run_parent_first_sampling([bind(r) for r in data['parents']],
        [bind(r) for r in data['references']], args.output,
        model_path=data['model_path'], model_sha256=data['model_sha256'],
        feature_model_path=data['feature_model_path'], feature_model_sha256=data['feature_model_sha256'],
        target=data['target'], frames_per_parent=data['frames_per_parent'], seed=data['seed'],
        min_distance_A=data['min_distance_A'], volume_ratio=data['volume_ratio'],
        checkpoint_dir=args.checkpoint_dir, binding=binding)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()

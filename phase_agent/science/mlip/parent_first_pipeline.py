"""Compute-node continuation from parent relaxation through DFT dispatch.

The caller must authenticate the input archive and approve the operation.
The existing Slurm dispatcher records each submission and blocks ambiguous replay.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _save(path, data):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def run_parent_first_pipeline(input_dir, results_dir, checkpoint_dir, operation_dir,
                              *, sampling=None, prepare=None, dispatch=None, validate=None, publish=None):
    """Run or resume with verified inputs; never infer an interrupted submission."""
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('parent_first_pipeline_requires_compute_job')
    from phase_agent.tools.remote.transfer_bundle import _archive_lock

    operation = Path(operation_dir).resolve()
    lock_root = operation.parent / ('.' + operation.name + '.pipeline-lock')
    lock_root.mkdir(parents=True, exist_ok=True)
    with _archive_lock(lock_root):
        return _run_locked(input_dir, results_dir, checkpoint_dir, operation_dir,
                           sampling=sampling, prepare=prepare, dispatch=dispatch, validate=validate, publish=publish)


def _run_locked(input_dir, results_dir, checkpoint_dir, operation_dir,
                *, sampling, prepare, dispatch, validate, publish):
    from phase_agent.science.mlip.parent_first_sampling import main as sampling_main
    from phase_agent.tools.remote.submission_dispatcher import run as dispatch_jobs
    from phase_agent.tools.workflows.parent_first_dft_handoff import prepare_parent_first_dft_handoff
    from phase_agent.tools.workflows.parent_first_result import validate_parent_first_result
    from phase_agent.tools.workflows.parent_first_return import prepare_parent_first_return

    sampling = sampling or sampling_main
    prepare = prepare or prepare_parent_first_dft_handoff
    dispatch = dispatch or dispatch_jobs
    validate = validate or validate_parent_first_result
    publish = publish or prepare_parent_first_return
    source = Path(input_dir).resolve()
    result = Path(results_dir).resolve()
    checkpoint = Path(checkpoint_dir).resolve()
    operation = Path(operation_dir).resolve()
    manifest = source / 'input_manifest.json'
    digest = _sha(manifest)
    if not json.loads(manifest.read_text(encoding='utf8')).get('dft_settings'):
        raise ValueError('parent_first_pipeline_dft_settings_missing')
    operation.parent.mkdir(parents=True, exist_ok=True)
    state_path = operation.parent / (operation.name + '.pipeline.json')
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding='utf8'))
        if state.get('input_manifest_sha256') != digest:
            raise ValueError('parent_first_pipeline_input_changed')
    else:
        if operation.exists():
            raise ValueError('parent_first_pipeline_unregistered_operation')
        state = {'schema':'parent-first-pipeline-v1', 'input_manifest_sha256':digest,
                 'status':'sampling'}
        _save(state_path, state)
    if not result.exists():
        if state['status'] != 'sampling':
            raise ValueError('parent_first_pipeline_sampling_result_missing')
        sampling(['--input', str(manifest), '--output', str(result), '--checkpoint-dir', str(checkpoint)])
    checked = validate(source, result)
    if checked.get('status') != 'verified' or checked['binding'].get('input_manifest_sha256') != digest:
        raise ValueError('parent_first_pipeline_sampling_invalid')
    if state.get('sampling_result_sha256') and state['sampling_result_sha256'] != checked['result_sha256']:
        raise ValueError('parent_first_pipeline_sampling_changed')
    if state['status'] == 'sampling':
        if operation.exists():
            raise ValueError('parent_first_pipeline_preparation_requires_review')
        handoff = prepare(source, result, operation)
        archive = Path(handoff['handoff']['upload_archive']).resolve()
        state.update(status='dft_prepared', sampling_result_sha256=checked['result_sha256'],
                     dft_manifest=handoff['dft_manifest'], upload_archive=str(archive),
                     upload_archive_sha256=_sha(archive), dispatch_directory=str(archive.parent),
                     task_count=handoff['task_count'])
        _save(state_path, state)
    archive = Path(state['upload_archive'])
    if not archive.is_file() or _sha(archive) != state['upload_archive_sha256']:
        raise ValueError('parent_first_pipeline_dft_archive_changed')
    if state['status'] not in {'dft_prepared', 'dispatching', 'submitted'}:
        raise ValueError('parent_first_pipeline_state_invalid')
    state['status'] = 'dispatching'
    _save(state_path, state)
    dispatch(state['dispatch_directory'])
    state['status'] = 'submitted'
    _save(state_path, state)
    dispatch_root = Path(state['dispatch_directory'])
    if not dispatch_root.name.endswith('_input'):
        raise ValueError('parent_first_pipeline_return_directory_invalid')
    return_dir = dispatch_root.with_name(dispatch_root.name[:-6] + '_output')
    provenance = return_dir / 'parent-first-provenance.results.zip'
    if provenance.exists():
        if state.get('provenance_sha256') != _sha(provenance):
            raise ValueError('parent_first_pipeline_return_requires_review')
    else:
        publish(source, result, Path(state['dft_manifest']).parent, dispatch_root, provenance)
        state['provenance_sha256'] = _sha(provenance)
        _save(state_path, state)
    return {'status':'submitted', 'task_count':state['task_count'], 'state_path':str(state_path),
            'dispatch_directory':state['dispatch_directory'], 'provenance_archive':str(provenance)}


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--checkpoint-dir', required=True)
    parser.add_argument('--operation-dir', required=True)
    parser.add_argument('--status-path')
    args = parser.parse_args(argv)
    outcome = run_parent_first_pipeline(args.input_dir, args.results_dir, args.checkpoint_dir, args.operation_dir)
    if args.status_path:
        path = Path(args.status_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _save(path, outcome)
    print(json.dumps(outcome, ensure_ascii=False))


if __name__ == '__main__':
    main()

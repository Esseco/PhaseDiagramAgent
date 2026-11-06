"""Prepare deterministic DFT inputs for a verified parent-first DIRECT result.

This function does not register project state, submit jobs, or approve work.
Its caller must bind the input manifest to the approved action before invoking it.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_parent_first_dft(input_dir, result_dir, output_dir, settings):
    """Verify all selected frames and prepare static inputs in deterministic slots."""
    from phase_agent.science.dft.prepare_pycode_relax import prepare_pycode_relax
    from phase_agent.tools.remote.integrity import payload_checksum
    from phase_agent.persistence.ledger.phase_data_manager import _stable_id
    from phase_agent.science.dft.parent_first_settings import validate_parent_first_dft_settings
    from phase_agent.tools.workflows.parent_first_result import validate_parent_first_result

    validate_parent_first_dft_settings(settings)
    checked = validate_parent_first_result(input_dir, result_dir)
    manifest_path = Path(input_dir) / 'input_manifest.json'
    if (_sha(manifest_path) != checked['binding'].get('input_manifest_sha256')
            or json.loads(manifest_path.read_text(encoding='utf8')).get('dft_settings') != settings):
        raise ValueError('parent_first_dft_settings_binding_invalid')
    rows = checked['selected']
    if checked['status'] != 'verified' or len(rows) != checked['target']:
        raise ValueError('parent_first_dft_selection_invalid')
    for row in rows:
        path = Path(row['structure_path'])
        if not path.is_file() or _sha(path) != row['structure_sha256']:
            raise ValueError('parent_first_dft_frame_changed')
    out = Path(output_dir)
    if out.exists():
        raise FileExistsError('parent_first_dft_exists')
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.parent-first-dft-', dir=out.parent) as scratch:
        staging = Path(scratch) / 'dft'
        staging.mkdir()
        tasks = []
        batch_id = 'parent-first-' + checked['result_sha256'][:16]
        for index, row in enumerate(rows):
            structure = Path(row['structure_path'])
            if _sha(structure) != row['structure_sha256']:
                raise ValueError('parent_first_dft_frame_changed')
            key = ':'.join((settings['config_version'], batch_id, str(index),
                            row['structure_sha256'], 'DFT_SINGLE_POINT'))
            task_id = 'DFT-' + hashlib.sha256(key.encode()).hexdigest()[:12]
            arrangement = {'parent_first_sha256':row['structure_sha256'],
                           'parent_record_id':row['parent_record_id']}
            structure_id = _stable_id('S', {'branch_id':row['branch_id'], 'arrangement':arrangement})
            directory = f'{index:05d}-{task_id}'
            task_dir = staging / directory
            task_dir.mkdir()
            task = {'task_id': task_id, 'task_key': key, 'batch_id': batch_id,
                    'stage': 'dft_single_point', 'status': 'pending',
                    'config_version': settings['config_version'], 'model_version': settings['model_version'],
                    'parent_decision_id': settings['approval_record_id'],
                    'input_structure_sha256': row['structure_sha256'], 'structure_path': 'initial_structure.vasp',
                    'structure_id': structure_id, 'object_id': structure_id,
                    'arrangement': arrangement,
                    'parent_record_id': row['parent_record_id'], 'branch_id': row['branch_id'],
                    'parameters': settings['parameters'], 'protocol_version': 1,
                    'input_file_version': 'task-json-v1', 'sampling_result_sha256': checked['result_sha256'],
                    'sampling_input_sha256': checked['binding']['input_manifest_sha256']}
            prepared = prepare_pycode_relax({**task, 'structure_path': str(structure),
                                           'work_directory': str(task_dir)},
                                          manager=SimpleNamespace(data={}))
            if (prepared.get('generator') != 'atomate' or prepared.get('backend') != 'pycode_atomate2_static'
                    or not (task_dir / 'initial_structure.vasp').is_file()
                    or _sha(task_dir / 'initial_structure.vasp') != row['structure_sha256']):
                raise ValueError('parent_first_dft_preparation_invalid')
            task['atomate'] = {**prepared, 'calculation_directory': '.'}
            task['task_checksum'] = payload_checksum(task)
            (task_dir / 'task.json').write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding='utf8')
            tasks.append({'task_id': task_id, 'task_key': key, 'directory': directory,
                          'task_checksum': task['task_checksum'], 'input_structure_sha256': row['structure_sha256']})
        receipt = {'schema': 'parent-first-dft-prepared-v1', 'status': 'prepared', 'submitted': False,
                   'batch_id': batch_id, 'binding': checked['binding'], 'sampling_result_sha256': checked['result_sha256'],
                   'config_version': settings['config_version'], 'model_version': settings['model_version'],
                   'approval_record_id': settings['approval_record_id'], 'task_count': len(tasks), 'tasks': tasks}
        (staging / 'dft_manifest.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf8')
        staging.rename(out)
    return receipt

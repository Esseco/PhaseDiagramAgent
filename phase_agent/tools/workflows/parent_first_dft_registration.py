"""Build a verified state update for remotely prepared parent-first DFT tasks.

The Agent must invoke this within its normal state lock and commit protocol.
It performs no I/O mutation or job submission.
"""
from __future__ import annotations

from copy import copy, deepcopy
import hashlib
import json
from pathlib import Path


def register_parent_first_dft_tasks(state, input_dir, result_dir, dft_dir, *, results_dir=None):
    from phase_agent.tools.remote.integrity import payload_checksum
    from phase_agent.persistence.ledger.phase_data_manager import _stable_id
    from phase_agent.science.dft.parent_first_settings import validate_parent_first_dft_settings
    from phase_agent.tools.workflows.parent_first_result import validate_parent_first_result

    if not state.get('action_records'):
        raise ValueError('parent_first_dft_approval_invalid')
    manifest_path = Path(input_dir) / 'input_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    settings = manifest.get('dft_settings') or {}
    validate_parent_first_dft_settings(settings)
    approval = next((r for r in state['action_records']
                     if r.get('record_id') == settings.get('approval_record_id')), {})
    action = approval.get('final_action') or {}
    params = action.get('parameters') or {}
    if ((approval.get('human_feedback') or {}).get('decision') != 'approve'
            or approval.get('status') not in {'prepared', 'completed', 'tasks_prepared'}
            or action.get('operation') != 'prepare_parent_first_recovery'
            or action.get('tool') != 'prepare_local_batch_files'
            or params.get('mode') != 'parent_first_recovery_inputs'
            or params.get('dft_parameters') != settings.get('parameters')
            or settings.get('config_version') != state.get('confirmed_config_version')
            or settings.get('model_version') != state.get('active_model_version')):
        raise ValueError('parent_first_dft_approval_invalid')
    expected = {'scope_id': 'scope_id', 'round_evidence_digest': 'round_evidence_digest',
                'direct_target': 'target', 'frames_per_parent': 'frames_per_parent', 'seed': 'seed',
                'min_distance_A': 'min_distance_A', 'volume_ratio': 'volume_ratio'}
    if (any(params.get(left) != manifest.get(right) for left, right in expected.items())
            or params.get('parent_record_ids') != [p.get('record_id') for p in manifest.get('parents') or []]):
        raise ValueError('parent_first_dft_approval_binding_invalid')
    batches = [(key, row) for key, row in (state.get('parent_first_recovery_batches') or {}).items()
               if row.get('approval_record_id') == settings['approval_record_id']]
    if (len(batches) != 1 or batches[0][1].get('scope_id') != manifest.get('scope_id')
            or batches[0][1].get('round_evidence_digest') != manifest.get('round_evidence_digest')
            or batches[0][1].get('target') != manifest.get('target')):
        raise ValueError('parent_first_dft_registered_batch_invalid')
    batch_key = batches[0][0]
    checked = validate_parent_first_result(input_dir, result_dir)
    saved = json.loads((Path(dft_dir) / 'dft_manifest.json').read_text(encoding='utf8'))
    rows = checked['selected']
    if (checked.get('status') != 'verified' or len(rows) != checked.get('target')
            or saved.get('schema') != 'parent-first-dft-prepared-v1'
            or saved.get('binding') != checked['binding']
            or saved.get('sampling_result_sha256') != checked['result_sha256']
            or saved.get('task_count') != len(rows) or len(saved.get('tasks') or []) != len(rows)
            or any(saved.get(k) != settings.get(k) for k in ('config_version','model_version','approval_record_id'))):
        raise ValueError('parent_first_dft_manifest_invalid')
    batch_id = 'parent-first-' + checked['result_sha256'][:16]
    if saved.get('batch_id') != batch_id:
        raise ValueError('parent_first_dft_manifest_invalid')
    existing = {t.get('task_id'):t for t in state.get('tasks') or []}
    registered = []
    for index, (entry, row) in enumerate(zip(saved['tasks'], rows)):
        key = ':'.join((settings['config_version'], batch_id, str(index), row['structure_sha256'], 'DFT_SINGLE_POINT'))
        task_id = 'DFT-' + hashlib.sha256(key.encode()).hexdigest()[:12]
        arrangement = {'parent_first_sha256':row['structure_sha256'],
                       'parent_record_id':row['parent_record_id']}
        structure_id = _stable_id('S', {'branch_id':row['branch_id'], 'arrangement':arrangement})
        folder = f'{index:05d}-{task_id}'
        if entry.get('directory') != folder or entry.get('task_id') != task_id or entry.get('task_key') != key:
            raise ValueError('parent_first_dft_task_invalid')
        path = Path(dft_dir) / folder / 'task.json'
        task = json.loads(path.read_text(encoding='utf8'))
        checksum = task.pop('task_checksum', None)
        expected_fields = {'task_id':task_id, 'task_key':key, 'batch_id':batch_id,
            'stage':'dft_single_point', 'status':'pending', 'config_version':settings['config_version'],
            'model_version':settings['model_version'], 'parent_decision_id':settings['approval_record_id'],
            'input_structure_sha256':row['structure_sha256'], 'structure_path':'initial_structure.vasp',
            'structure_id':structure_id, 'object_id':structure_id,
            'arrangement':arrangement,
            'parent_record_id':row['parent_record_id'], 'branch_id':row['branch_id'],
            'parameters':settings['parameters'], 'sampling_result_sha256':checked['result_sha256'],
            'sampling_input_sha256':checked['binding']['input_manifest_sha256'],
            'protocol_version':1, 'input_file_version':'task-json-v1'}
        if (checksum != payload_checksum(task) or checksum != entry.get('task_checksum')
                or entry.get('input_structure_sha256') != row['structure_sha256']
                or (task.get('atomate') or {}).get('backend') != 'pycode_atomate2_static'
                or any(task.get(k) != v for k, v in expected_fields.items())):
            raise ValueError('parent_first_dft_task_invalid')
        registered_task = {**task, 'task_checksum':checksum, 'structure_path':row['structure_path'],
                           'upload_operation_id':batch_id, 'input_path':str(path)}
        if results_dir is not None:
            registered_task['result_path'] = str(Path(results_dir)/folder/'result.json')
        prior = existing.get(task_id)
        if prior:
            stable = ('task_id','task_key','task_checksum','input_structure_sha256','config_version',
                      'model_version','parent_decision_id','upload_operation_id','structure_path','input_path')
            result_matches = (prior.get('result_path') == registered_task.get('result_path') or
                              prior.get('calculation_result_path') == registered_task.get('result_path'))
            if any(prior.get(k) != registered_task.get(k) for k in stable) or not result_matches:
                raise ValueError('parent_first_dft_registration_conflict')
        registered.append(registered_task)
    present = sum(row['task_id'] in existing for row in registered)
    if present:
        if present != len(registered):
            raise ValueError('parent_first_dft_registration_incomplete')
        return {'status':'already_registered', 'state':deepcopy(state), 'tasks':registered, 'submitted':False}
    current = deepcopy(state)
    current.setdefault('tasks', []).extend(deepcopy(registered))
    current.setdefault('pending_tasks', []).extend(deepcopy(registered))
    current['parent_first_recovery_batches'][batch_key].update(
        status='waiting_dft_results', dft_task_ids=[row['task_id'] for row in registered],
        sampling_result_sha256=checked['result_sha256'], selection_dir=str(Path(result_dir).resolve()),
        dft_input_dir=str(Path(dft_dir).resolve()))
    return {'status':'registered', 'state':current, 'tasks':registered, 'submitted':False}


def stage_parent_first_structures(manager, tasks):
    """Return a private ledger copy; caller commits it with state atomically."""
    shadow = copy(manager)
    shadow.data = deepcopy(manager.data)
    for task in tasks:
        structure_id = shadow.add_structure(branch_id=task['branch_id'], arrangement=task['arrangement'],
            source_path=task['structure_path'], metadata={'parent_first_input_sha256':task['input_structure_sha256'],
                'perturbation_recovery_batch':task['upload_operation_id']})
        if structure_id != task['structure_id']:
            raise ValueError('parent_first_structure_identity_mismatch')
        row = shadow.data['structures'][structure_id]
        if row.get('source_path') != task['structure_path']:
            raise ValueError('parent_first_structure_source_conflict')
    return shadow

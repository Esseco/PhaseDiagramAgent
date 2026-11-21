"""Prepare approved perturbation frames as bounded MH-1 upload tasks."""
from __future__ import annotations

from copy import copy, deepcopy
import hashlib
import json
from pathlib import Path

from ase.io import read
from phase_agent.configuration.defaults.default_slurm_cluster_config import default_slurm_cluster_config
from phase_agent.science.mlip.slurm_executor import create_mlip_task_preparer
from phase_agent.tools.budget.check_budget import check_budget
from phase_agent.tools.budget.estimate_stage_cost import estimate_stage_cost
from phase_agent.tools.budget.reserve_budget import reserve_owned_budget
from phase_agent.tools.remote.manual_upload_runner import ManualUploadBatchRunner
from phase_agent.tools.workflows.perturbation_staging import prepare_perturbation_inputs


def prepare_perturbation_relax(*, action, context):
    """Prepare local inputs only; never submit or replace the active model."""
    approval = context.get('approval_record_id')
    manager = context.get('manager')
    config = deepcopy(context.get('effective_config') or {})
    source = context.get('event_state') or {}
    parameters = deepcopy(action.get('parameters') or {})
    root = config.get('upload_batches_directory')
    command = ((config.get('supercomputer') or {}).get('worker') or {}).get('command') or []
    if not approval or manager is None or not root or not command or not config.get('budgets') or not config.get('ledger_path'):
        return {'status': 'not_configured', 'state': source, 'submitted': False,
                'reason': 'perturbation_approval_manager_or_upload_config_missing'}
    existing = source.get('perturbation_recovery_batch') or {}
    if existing:
        if existing.get('approval_record_id') == approval and existing.get('scope_id') == parameters.get('scope_id'):
            manifest = Path(existing.get('manifest_path') or '')
            archive = Path(existing.get('upload_archive') or '')
            ids = existing.get('task_ids') or []
            known = {t.get('task_id') for t in source.get('tasks') or []}
            if (manifest.is_file() and hashlib.sha256(manifest.read_bytes()).hexdigest() == existing.get('manifest_sha256')
                and archive.is_file() and ids and set(ids).issubset(known)):
                return {'status': 'already_prepared', 'state': source, 'submitted': False,
                        'perturbation_recovery_batch': existing}
            return {'status': 'not_configured', 'state': source, 'submitted': False,
                    'reason': 'perturbation_existing_inputs_require_recovery'}
        raise ValueError('perturbation_existing_batch_requires_recovery')
    parameters['confirmed'] = True
    key = hashlib.sha256(json.dumps({'approval': approval, 'parameters': parameters},
                                    sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    target = Path(root) / 'perturbation_inputs' / key[:20]
    staged = prepare_perturbation_inputs(source, manager.data, parameters, target)
    manifest_path = Path(staged['manifest_path'])
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    # Preserve all parent provenance while calculating byte-identical frames once.
    unique, duplicates = {}, []
    for frame in manifest['frames']:
        if frame['sha256'] in unique:
            duplicates.append(frame)
        else:
            unique[frame['sha256']] = frame
    frame_rows = list(unique.values())
    if duplicates:
        manifest['frames'] = frame_rows
        manifest['duplicate_frames'] = duplicates
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')
    model = deepcopy(manifest['relax_model']['model'])
    model.update(version='mace-mh-1', sha256=manifest['relax_model']['sha256'])
    settings = deepcopy(model.get('relax_parameters') or {})
    config['mlip'] = model
    state = deepcopy(source)
    shadow = copy(manager)
    shadow.data = deepcopy(manager.data)
    planned = []
    for frame in frame_rows:
        path = (manifest_path.parent / frame['path']).resolve()
        if not path.is_relative_to(manifest_path.parent.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != frame['sha256']:
            raise ValueError('perturbation_input_frame_invalid')
        atom_count = len(read(path, index=0))
        cost = estimate_stage_cost('relax_and_feature', atom_count=atom_count, budgets=config['budgets'])['value']
        planned.append((frame, path, cost))
    checked = check_budget(state, {'stage': 'relax_and_feature', 'tasks': len(planned),
                                   'relative_cost': sum(row[2] for row in planned)}, config['budgets'])
    if not checked['allowed']:
        return {'status': 'budget_exhausted', 'state': source, 'submitted': False, 'reason': checked['reasons']}
    ids = []
    batch_id = key[:20]
    for frame, path, cost in planned:
        sid = shadow.add_structure(branch_id=frame['branch_id'],
            arrangement={'perturbation_sha256': frame['sha256'], 'parent_record_id': frame['parent_record_id']},
            source_path=path, metadata={'perturbation_recovery_batch': batch_id})
        task_key = 'perturbation:' + key + ':' + frame['sha256']
        if any(t.get('task_key') == task_key for t in state.get('tasks') or []):
            raise ValueError('perturbation_task_already_exists')
        reservation = reserve_owned_budget(state, task_key=task_key, stage='relax_and_feature',
            amount=cost, limits=config['budgets'], config_version=context.get('config_version'), model_version='mace-mh-1')
        if reservation['status'] != 'reserved':
            raise ValueError('perturbation_budget_reservation_failed')
        task_id = 'PERT-' + hashlib.sha256(task_key.encode()).hexdigest()[:16]
        task = {'task_id': task_id, 'task_key': task_key, 'stage': 'relax_and_feature', 'status': 'pending',
                'structure_id': sid, 'object_id': sid, 'branch_id': frame['branch_id'], 'structure_path': str(path),
                'input_structure_sha256': frame['sha256'], 'perturbation_batch_id': batch_id,
                'model_version': 'mace-mh-1', 'model_sha256': model['sha256'],
                'upload_epoch_model_version': manifest['failed_model_version'],
                'config_version': context.get('config_version'), 'planned_relative_cost': cost,
                'upload_operation_id': batch_id, 'parent_decision_id': approval, 'parameters': settings}
        state.setdefault('tasks', []).append(task)
        state.setdefault('pending_tasks', []).append(task)
        ids.append(task_id)
    # The old generation gate contains branch structures, not these verified
    # perturbation frames. Extend only this private preparation view, then restore
    # the original gate before publishing state; never bypass an unresolved gate.
    original_gate = deepcopy(state.get('dedup_gate'))
    if original_gate and original_gate.get('status') == 'ready' and original_gate.get('valid_structure_ids'):
        state['dedup_gate'] = {**original_gate, 'valid_structure_ids': sorted(
            set(original_gate['valid_structure_ids']) | {t['structure_id'] for t in state['tasks'] if t.get('task_id') in ids})}
    runner = ManualUploadBatchRunner(root, task_ids=ids, worker_command=command,
        stage_batch_sizes=((config.get('supercomputer') or {}).get('batch_sizes') or {}),
        stage_profiles=default_slurm_cluster_config(),
        task_preparer=create_mlip_task_preparer(shadow, context.get('phase_references') or {}, config))
    batches = []
    while True:
        result = runner.prepare(state, _owned=True, _defer_index=True, _defer_archive=True)
        state = result['state']
        if result['status'] == 'no_tasks':
            break
        if result['status'] != 'prepared':
            raise ValueError('perturbation_upload_preparation_failed')
        batches.append(result['batch'])
    if original_gate is not None:
        state['dedup_gate'] = original_gate
    prepared = {t['task_id'] for t in state.get('tasks') or [] if t.get('input_path')}
    if not set(ids).issubset(prepared):
        raise ValueError('perturbation_upload_incomplete')
    state['perturbation_recovery_batch'] = {
        'status': 'waiting_results', 'batch_id': batch_id, 'task_ids': ids,
        'scope_id': manifest['scope_id'], 'round_evidence_digest': manifest['round_evidence_digest'],
        'manifest_path': str(manifest_path), 'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'approval_record_id': approval, 'failed_model_version': manifest['failed_model_version'],
        'model_version': 'mace-mh-1', 'model_sha256': model['sha256']}
    # Package only this operation's newly prepared batches, never unrelated jobs.
    from phase_agent.tools.remote.submission_handoff import prepare_batch_handoff
    batch_ids = {b['batch_id'] for b in batches}
    scoped = dict(state)
    scoped['slurm_batches'] = [b for b in state.get('slurm_batches') or [] if b.get('batch_id') in batch_ids]
    scoped['tasks'] = [t for t in state.get('tasks') or [] if t.get('task_id') in ids]
    if not scoped['slurm_batches']:
        raise ValueError('perturbation_handoff_batch_missing')
    handoff = prepare_batch_handoff(scoped, scoped['slurm_batches'][0])
    if not handoff or not Path(handoff.get('upload_archive') or '').is_file():
        raise ValueError('perturbation_handoff_archive_missing')
    state['perturbation_recovery_batch']['upload_archive'] = handoff['upload_archive']
    from phase_agent.tools.remote.submission_layout import write_submission_index
    write_submission_index(root, state)
    # Persist derived structure identities before the task state can be saved.
    shadow.save(config['ledger_path'])
    manager.data = shadow.data
    for attr in ('_ledger_path', '_ledger_identity', '_ledger_data_digest'):
        if hasattr(shadow, attr):
            setattr(manager, attr, getattr(shadow, attr))
    return {'status': 'prepared', 'state': state, 'submitted': False,
            'task_count': len(ids), 'batch_count': len(batches), 'handoff': handoff,
            'perturbation_recovery_batch': state['perturbation_recovery_batch']}

"""Verify an approved, failed legacy perturbation staging without changing files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect_abandoned_perturbation(state, identity):
    """Bind a failed legacy action to its exact local frames, never to HPC work."""
    invocation = identity.get('invocation_id')
    records = [r for r in state.get('action_records') or [] if r.get('record_id') == invocation]
    if len(records) != 1:
        raise ValueError('abandoned_staging_record_invalid')
    record = records[0]
    action = record.get('final_action') or {}
    execution = record.get('execution_result') or {}
    from phase_agent.policies.file_approval import proposal_hash
    if (identity.get('tool') != action.get('tool') or identity.get('action_hash') != proposal_hash(action)):
        raise ValueError('abandoned_staging_action_identity_invalid')
    if (record.get('status') != 'failed' or (record.get('human_feedback') or {}).get('decision') != 'approve'
            or action.get('operation') != 'prepare_perturbation_recovery'
            or (action.get('parameters') or {}).get('mode') != 'perturbation_recovery_inputs'
            or execution.get('error') != 'ValueError: perturbation_upload_incomplete'):
        raise ValueError('abandoned_staging_original_action_invalid')
    params = dict(action['parameters'])
    params['confirmed'] = True
    key = hashlib.sha256(json.dumps({'approval': invocation, 'parameters': params},
                                   sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]
    storage = (state.get('confirmed_config') or {}).get('storage') or {}
    root = Path(storage.get('workspace_root') or '').resolve()
    upload = Path((storage.get('paths') or {}).get('upload_batches') or '')
    upload = (upload if upload.is_absolute() else root / upload).resolve()
    if not upload.is_relative_to(root):
        raise ValueError('abandoned_staging_storage_invalid')
    directory = (upload / 'perturbation_inputs' / key).resolve()
    if not directory.is_relative_to(upload):
        raise ValueError('abandoned_staging_path_invalid')
    manifest_path = directory / 'manifest.json'
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    expected_params = {'frames_per_parent': params.get('frames_per_parent', 36),
                       'seed': params.get('seed', 42), 'max_det_H': params.get('max_det_H'),
                       'min_distance_A': params.get('min_distance_A', 1.0),
                       'volume_ratio': list(params.get('volume_ratio', (0.75, 1.25)))}
    if (manifest.get('schema') != 'perturbation-recovery-inputs-v1'
            or manifest.get('status') != 'structures_prepared' or manifest.get('submitted') is not False
            or manifest.get('scope_id') != params.get('scope_id')
            or manifest.get('round_evidence_digest') != params.get('round_evidence_digest')
            or manifest.get('failed_model_version') != (execution.get('operation_outcome') or {}).get('model_version')
            or manifest.get('parameters') != expected_params):
        raise ValueError('abandoned_staging_manifest_binding_invalid')
    parents = manifest.get('parents') or []
    if [p.get('record_id') for p in parents] != params.get('parent_record_ids'):
        raise ValueError('abandoned_staging_parents_invalid')
    for parent in parents:
        path = Path(parent.get('structure_path') or '')
        if not path.is_file() or _sha(path) != parent.get('structure_sha256'):
            raise ValueError('abandoned_staging_parent_changed')
    frames = manifest.get('frames') or []
    if not frames or len({f.get('path') for f in frames}) != len(frames):
        raise ValueError('abandoned_staging_frames_invalid')
    parent_by_id = {p['record_id']: p for p in parents}
    for frame in frames:
        parent = parent_by_id.get(frame.get('parent_record_id'))
        path = (directory / (frame.get('path') or '')).resolve()
        if (not parent or frame.get('parent_sha256') != parent.get('structure_sha256')
                or not path.is_relative_to(directory) or not path.is_file() or _sha(path) != frame.get('sha256')):
            raise ValueError('abandoned_staging_frame_invalid')
    if any(t.get('perturbation_batch_id') == key or t.get('upload_operation_id') == key
           for t in state.get('tasks') or []):
        raise ValueError('abandoned_staging_has_registered_tasks')
    if any(b.get('upload_operation_id') == key for b in state.get('slurm_batches') or []):
        raise ValueError('abandoned_staging_has_registered_batches')
    # A failed handler may have written unregistered task inputs before the
    # exception. Those require their own recovery, not abandonment as staged-only.
    for task_path in upload.rglob('task.json'):
        try:
            task = json.loads(task_path.read_text(encoding='utf8'))
        except (OSError, ValueError) as error:
            raise ValueError('abandoned_staging_task_inventory_unreadable') from error
        if (str(task.get('task_key') or '').startswith('perturbation:' + key + ':')
                or task.get('perturbation_batch_id') == key or task.get('upload_operation_id') == key):
            raise ValueError('abandoned_staging_has_unregistered_task_inputs')
    decision = (state.get('effective_decisions') or {}).get(action.get('task_key')) or {}
    if decision.get('status') in {'reserved', 'reconciliation_required'}:
        raise ValueError('abandoned_staging_decision_requires_settlement')
    return {'kind':'abandoned_legacy_perturbation_staging', 'directory':str(directory),
            'manifest_sha256':hashlib.sha256(raw).hexdigest(), 'parent_count':len(parents),
            'frame_count':len(frames), 'batch_key':key, 'submitted':False}

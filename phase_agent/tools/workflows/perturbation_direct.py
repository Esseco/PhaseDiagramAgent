"""Validate MH-1 relaxed perturbation frames before sharing DIRECT sampling."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from phase_agent.science.structures.tm_order import tm_order_preserved as _tm_order_preserved


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def perturbation_candidates(state, batch, *, scope_id, round_digest):
    """Return only proven normal final frames; missing or altered inputs fail closed."""
    if (batch.get('status') != 'results_received' or batch.get('scope_id') != scope_id
        or batch.get('round_evidence_digest') != round_digest
        or not batch.get('approval_record_id')):
        raise ValueError('perturbation_batch_binding_invalid')
    manifest_path = Path(batch.get('manifest_path') or '')
    if not manifest_path.is_file() or _digest(manifest_path) != batch.get('manifest_sha256'):
        raise ValueError('perturbation_manifest_invalid')
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    model = manifest.get('relax_model') or {}
    if (manifest.get('schema') != 'perturbation-recovery-inputs-v1'
        or manifest.get('scope_id') != scope_id or manifest.get('round_evidence_digest') != round_digest
        or manifest.get('failed_model_version') != state.get('active_model_version')
        or model.get('version') != 'mace-mh-1' or not isinstance(model.get('sha256'), str)
        or not re.fullmatch(r'[a-fA-F0-9]{64}', model['sha256'])):
        raise ValueError('perturbation_manifest_binding_invalid')
    frames = manifest.get('frames') or []
    by_hash = {r.get('sha256'): r for r in frames}
    if not frames or len(by_hash) != len(frames) or None in by_hash:
        raise ValueError('perturbation_inputs_duplicated_or_missing')
    ids = batch.get('task_ids') or []
    tasks = {t.get('task_id'): t for t in state.get('tasks') or []}
    if not ids or len(set(ids)) != len(ids) or len(ids) != len(frames) or any(i not in tasks for i in ids):
        raise ValueError('perturbation_task_inventory_invalid')
    rows, excluded = [], {'failed': 0, 'extrapolation': 0, 'invalid_geometry': 0}
    used = set()
    from ase.io import read
    from Process_MLIP.nepkit_sampling import _valid
    for tid in ids:
        task = tasks[tid]
        key = task.get('input_structure_sha256')
        frame = by_hash.get(key)
        if (not frame or key in used or task.get('model_version') != 'mace-mh-1'
            or task.get('perturbation_batch_id') != batch.get('batch_id')
            or task.get('model_sha256') != model['sha256']):
            raise ValueError('perturbation_task_binding_invalid')
        used.add(key)
        source = (manifest_path.parent / frame['path']).resolve()
        if not source.is_relative_to(manifest_path.parent.resolve()) or not source.is_file() or _digest(source) != key:
            raise ValueError('perturbation_input_frame_invalid')
        if task.get('status') not in {'completed', 'failed'}:
            raise ValueError('perturbation_tasks_incomplete')
        if task.get('status') == 'failed':
            excluded['failed'] += 1
            continue
        output = task.get('outputs') or {}
        if output.get('verified_model_sha256') != model['sha256']:
            raise ValueError('perturbation_executed_model_unverified')
        monitor = output.get('extrapolation_monitor') or {}
        score = monitor.get('final_feature_difference')
        if (monitor.get('extrapolation_failure') is not False or monitor.get('failures') != []
            or isinstance(score, bool) or not isinstance(score, (int, float))
            or not math.isfinite(score) or score < 0 or score > 1):
            excluded['extrapolation'] += 1
            continue
        path = Path(output.get('structure_path') or '')
        checksum = output.get('structure_checksum')
        if not path.is_file() or not checksum or _digest(path) != checksum:
            raise ValueError('perturbation_final_frame_invalid')
        original, final = read(source, index=0), read(path, index=0)
        params = manifest.get('parameters') or {}
        if (not final.pbc.all() or original.get_chemical_symbols() != final.get_chemical_symbols()
            or not _tm_order_preserved(original, final)
            or not _valid(final, abs(original.get_volume()), params.get('min_distance_A', 1.0),
                          params.get('volume_ratio', (0.75, 1.25)))):
            excluded['invalid_geometry'] += 1
            continue
        rows.append({'candidate_id': 'PERTURB-' + tid, 'source_task_id': tid,
                     'structure_id': task.get('structure_id'), 'branch_id': frame.get('branch_id'),
                     'structure_path': str(path), 'structure_sha256': checksum,
                     'model_version': 'mace-mh-1', 'final_feature_difference': score,
                     'composition': {symbol: final.get_chemical_symbols().count(symbol)
                                     for symbol in sorted(set(final.get_chemical_symbols()))},
                     'atom_count': len(final), 'phase': frame.get('phase'),
                     'extrapolation_failure': False, 'candidate_origin': 'perturbation_recovery',
                     'qbc': {'status': 'not_available_exact_current_geometry'}})
    if not rows:
        raise ValueError('perturbation_no_valid_relaxed_frames')
    return rows, excluded


def approved_perturbation_direct_target(state):
    """Return the explicitly approved target; old approvals retain their behavior."""
    batch = state.get('perturbation_recovery_batch') or {}
    record = next((r for r in state.get('action_records') or []
                   if r.get('record_id') == batch.get('approval_record_id')), None) or {}
    approved = record.get('final_action') or {}
    if ((record.get('human_feedback') or {}).get('decision') != 'approve'
        or approved.get('operation') != 'prepare_perturbation_recovery'):
        return None
    target = (approved.get('parameters') or {}).get('direct_target')
    if target is not None and (type(target) is not int or not 1 <= target <= 100):
        raise ValueError('approved_perturbation_direct_target_invalid')
    return target


def previously_approved_direct(state, action):
    """Only the exact reviewed perturbation batch authorizes its DIRECT stage."""
    batch = state.get('perturbation_recovery_batch') or {}
    params = action.get('parameters') or {}
    target = params.get('target', 100)
    if (batch.get('status') != 'results_received' or action.get('operation') != 'sample_direct_extrapolation'
        or action.get('tool') != 'prepare_local_batch_files' or params.get('mode') != 'direct_extrapolation'
        or params.get('recovery_route') != 'perturbation' or params.get('scope_id') != batch.get('scope_id')
        or params.get('round_evidence_digest') != batch.get('round_evidence_digest')
        or type(target) is not int or not 1 <= target <= 100):
        return False
    record = next((r for r in state.get('action_records') or []
                   if r.get('record_id') == batch.get('approval_record_id')), None) or {}
    approved = record.get('final_action') or {}
    original = approved.get('parameters') or {}
    if ((record.get('human_feedback') or {}).get('decision') != 'approve'
        or record.get('status') not in {'prepared', 'completed', 'tasks_prepared'}
        or approved.get('operation') != 'prepare_perturbation_recovery'
        or original.get('scope_id') != batch.get('scope_id')
        or original.get('round_evidence_digest') != batch.get('round_evidence_digest')):
        return False
    approved_target = approved_perturbation_direct_target(state)
    if approved_target is not None and target != approved_target:
        return False
    try:
        perturbation_candidates(state, batch, scope_id=batch['scope_id'], round_digest=batch['round_evidence_digest'])
    except (ValueError, OSError, KeyError, TypeError):
        return False
    return True


def previously_approved_recovery_action(state, action):
    """Inherit route approval only for exact selected DFT frames or their training."""
    cycle = state.get('extrapolation_recovery_cycle') or {}
    if cycle.get('status') != 'awaiting_reviewed_dft_labels':
        return False
    if cycle.get('recovery_route') == 'parent_first_perturbation':
        from phase_agent.evidence.dft_failure_policy import unreviewed_failure_batches
        from phase_agent.evidence.extrapolation_recovery import recovery_label_facts
        from phase_agent.evidence.parent_first_recovery import valid_parent_first_cycle
        record = next((r for r in state.get('action_records') or []
                       if r.get('record_id') == cycle.get('recovery_approval_record_id')), {})
        approved = record.get('final_action') or {}
        original = approved.get('parameters') or {}
        batch = (state.get('parent_first_recovery_batches') or {}).get(cycle.get('batch_id')) or {}
        params = action.get('parameters') or {}
        return (valid_parent_first_cycle(state) and not unreviewed_failure_batches(state)
                and (record.get('human_feedback') or {}).get('decision') == 'approve'
                and record.get('status') in {'prepared', 'completed', 'tasks_prepared'}
                and approved.get('operation') == 'prepare_parent_first_recovery'
                and original.get('scope_id') == batch.get('scope_id')
                and original.get('round_evidence_digest') == batch.get('round_evidence_digest')
                and action.get('tool') == 'update_mlip' and action.get('operation') == 'prepare_mlip_training'
                and params.get('action') == 'RETRAIN_MLIP' and params.get('prepare_inputs_only') is True
                and recovery_label_facts(state).get('status') == 'complete')
    receipt = (state.get('direct_sampling_results') or {}).get(cycle.get('scope_id')) or {}
    from phase_agent.tools.workflows.direct_extrapolation_sampling import valid_direct_receipt
    if (not valid_direct_receipt(receipt) or receipt.get('source_digest') != cycle.get('direct_source_digest')
        or receipt.get('recovery_approval_record_id') != cycle.get('recovery_approval_record_id')):
        return False
    record = next((r for r in state.get('action_records') or []
                   if r.get('record_id') == cycle.get('recovery_approval_record_id')), None) or {}
    approved = record.get('final_action') or {}
    original = approved.get('parameters') or {}
    expected_operation = ('prepare_perturbation_recovery' if cycle.get('recovery_route') == 'perturbation'
                          else 'sample_direct_extrapolation')
    if ((record.get('human_feedback') or {}).get('decision') != 'approve'
        or record.get('status') not in {'prepared', 'completed', 'tasks_prepared'}
        or approved.get('operation') != expected_operation
        or original.get('scope_id') != cycle.get('scope_id')
        or original.get('round_evidence_digest') != receipt.get('round_evidence_digest')):
        return False
    if action.get('tool') == 'select_dft_candidates':
        decisions = (action.get('parameters') or {}).get('decisions') or []
        if not isinstance(decisions, list) or not decisions or any(
            not isinstance(row, dict) or row.get('action') not in {'DFT_SINGLE_POINT', 'DFT_RELAX'}
            for row in decisions):
            return False
        ids = {row.get('candidate_id') for row in decisions}
        if len(ids) != len(decisions):
            return False
        from phase_agent.tools.workflows.validate_dft_agent_decisions import _is_direct_recovery_selection
        return _is_direct_recovery_selection(state, ids)
    if action.get('tool') == 'update_mlip' and action.get('operation') == 'prepare_mlip_training':
        params = action.get('parameters') or {}
        from phase_agent.evidence.extrapolation_recovery import recovery_label_facts
        return (params.get('action') == 'RETRAIN_MLIP' and params.get('prepare_inputs_only') is True
                and recovery_label_facts(state).get('status') == 'complete')
    return False

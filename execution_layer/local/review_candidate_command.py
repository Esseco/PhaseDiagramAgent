"""Explicit candidate review commands; generic continue/approve never activates."""
import re
from pathlib import Path
from execution_layer.local.training_handoff import digest_file
from execution_layer.local.recover_remote_training import inspect_training_results
from execution_layer.workflows.create_model_update_handler import create_model_update_handler


def review_candidate_command(message, state):
    match = re.fullmatch(r'(激活候选|拒绝候选)\s+(\S+)\s+原因[：:]\s*(.+)', str(message).strip(), re.DOTALL)
    if not match:
        return None
    command, version, reason = match.groups()
    if not reason.strip():
        return {'state': state, 'reason': '请提供明确的审阅原因；未激活。'}
    candidate = (state.get('candidate_models') or {}).get(version)
    if not candidate:
        return {'state': state, 'reason': '候选版本不存在；未激活。'}
    if command == '激活候选':
        if candidate.get('status') == 'user_rejected':
            return {'state': state, 'reason': '该候选已被拒绝；需重新审阅，未激活。'}
        if state.get('active_model_version') == version:
            return {'state': state, 'reason': '该候选已经激活；未重复执行。'}
        if state.get('active_model_version') != candidate.get('old_model_version'):
            return {'state': state, 'reason': '当前旧模型版本已改变，需重新验证；未激活。'}
        jobs = [job for job in (state.get('remote_finetune_jobs') or {}).values()
                if (job.get('training_handoff') or {}).get('candidate_model_version') == version]
        if len(jobs) != 1:
            return {'state': state, 'reason': '候选训练来源不唯一；未激活。'}
        job = jobs[0]
        handoff = job['training_handoff']
        result = inspect_training_results(job)
        artifact = Path(job['directory']) / 'results' / ('validation-' + handoff['request_id'] + '.json')
        if (not result or result.get('fingerprint') != handoff.get('training_fingerprint')
                or not artifact.is_file() or digest_file(artifact) != handoff.get('validation_sha256')):
            return {'state': state, 'reason': '训练或验证回传已改变；请先继续重新回收，未激活。'}
    trigger = {'action': 'ACTIVATE_CANDIDATE_MODEL' if command == '激活候选' else 'REJECT_CANDIDATE_MODEL',
        'candidate_model_version': version,
        'user_approval_reason' if command == '激活候选' else 'user_rejection_reason': reason.strip()}
    result = create_model_update_handler()(trigger=trigger, state=state, manager=None, config={})
    updated = result['state']
    if result['status'] == 'activated':
        for job in updated.get('remote_finetune_jobs', {}).values():
            if (job.get('training_handoff') or {}).get('candidate_model_version') == version:
                job['activated'] = True
                job['training_handoff']['stage'] = 'activated'
    return {'state': updated, 'reason': f'候选 {version}：{result["status"]}。' +
            ('下一步说“继续”，审阅新模型结构刷新方案。' if result['status'] == 'activated' else '')}

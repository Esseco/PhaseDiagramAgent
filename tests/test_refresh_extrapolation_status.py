import hashlib
import json

from phase_agent.evidence.workflow_blockers import refresh_result_facts
from phase_agent.runtime.scientific_progress import project_progress


def case(tmp_path):
    output = tmp_path / 'results'
    output.mkdir()
    archive = output / 'part.results.zip'
    archive.write_bytes(b'bound result')
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    review = tmp_path / 'analysis' / 'epoch1' / 'extrapolation_review'
    review.mkdir(parents=True)
    report = review / 'test_外推证据说明.json'
    report.write_text(json.dumps({
        'source_archives_sha256': {archive.name: digest},
        'endpoint_screened': 1, 'endpoint_difference_gt_1': 1,
        'screened': [{'task_id':'t1', 'score':1.5, 'monitor_force_jump_available':False}],
    }), encoding='utf-8')
    state = {
        'analysis_output_directory': str(tmp_path / 'analysis'),
        'active_model_version': 'model-1',
        'model_refresh': {'status':'waiting_results','new_model_version':'model-1', 'task_ids':['t1','t2']},
        'tasks': [
            {'task_id':'t1', 'model_version':'model-1', 'status':'completed', 'converged':False,
             'result_disposition':'discarded_unconverged', 'outputs':{'optimizer_converged':False}},
            {'task_id':'t2', 'model_version':'model-1', 'status':'failed', 'error':'unconverged'},
        ],
        'slurm_batches': [{'task_ids':['t1','t2'], 'handoff':{'return_directory':str(output)}}],
    }
    return state, archive, report


def test_refresh_status_exposes_suspected_and_preserves_incomplete_evidence(tmp_path):
    state, _, _ = case(tmp_path)
    facts = refresh_result_facts(state)
    assert facts['status_counts'] == {'completed':1, 'failed':1}
    assert facts['discarded_unconverged_count'] == 1
    evidence = facts['extrapolation_review']
    assert evidence['suspected_count'] == 1
    assert evidence['coverage_complete'] is False
    assert evidence['force_jump_evidence_available'] is False
    progress = project_progress(state, tmp_path / 'state.json')
    assert progress['waiting_state']['kind'] == 'refresh_blocked'
    assert '疑似外推失败 1 项' in progress['waiting_state']['next_action']


def test_changed_archive_cannot_be_reported_as_zero_or_valid(tmp_path):
    state, archive, _ = case(tmp_path)
    archive.write_bytes(b'changed')
    evidence = refresh_result_facts(state)['extrapolation_review']
    assert evidence['status'] == 'unverified'
    assert evidence['suspected_count'] is None
    assert evidence['coverage_complete'] is False


def test_wrong_model_and_missing_report_do_not_confirm_evidence(tmp_path):
    state, _, report = case(tmp_path)
    state['tasks'][0]['model_version'] = 'other'
    assert refresh_result_facts(state)['extrapolation_review']['status'] == 'unverified'
    report.unlink()
    evidence = refresh_result_facts(state)['extrapolation_review']
    assert evidence['status'] == 'not_available'
    assert evidence['suspected_count'] is None


def test_refresh_gate_counts_only_accepted_results(tmp_path):
    from phase_agent.tools.workflows.model_refresh_gate import advance_refresh

    state, _, _ = case(tmp_path)
    state['tasks'].append({'task_id':'t3', 'model_version':'model-1', 'status':'completed',
                           'converged':True, 'outputs':{'optimizer_converged':True}})
    state['model_refresh']['task_ids'].append('t3')
    state['model_refresh']['wave'] = 0
    _, reason = advance_refresh(state)
    assert '有效结果1/3' in reason
    assert '未收敛已排除1' in reason


def test_agent_summary_keeps_bounded_extrapolation_evidence(tmp_path):
    from phase_agent.evidence.requests import summary_payload

    state, _, _ = case(tmp_path)
    facts = refresh_result_facts(state)
    result, _ = summary_payload({'state': {}, 'decision_context': {'refresh_result_facts':facts}})
    evidence = result['decision_context']['refresh_result_facts']['extrapolation_review']
    assert evidence['suspected_count'] == 1
    assert evidence['coverage_complete'] is False
    assert '不能判定' in evidence['instruction']

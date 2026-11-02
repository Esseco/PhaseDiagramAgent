from phase_agent.evidence.build_decision_context import build_decision_context
from phase_agent.tools.state.restart_failed_task import restart_failed_task
from phase_agent.configuration.defaults.default_budget_rules import default_budget_rules


def test_history_limit_does_not_remove_live_retry_permissions():
    tasks = [{'task_id': f'T{i}', 'stage': 'dft_single_point', 'status': 'failed'} for i in range(8)]
    context = build_decision_context({'tasks': tasks}, recent_limit=1)
    assert {t['task_id'] for t in context['retryable_tasks']} == {t['task_id'] for t in tasks}


def test_retry_drops_old_checksum_but_preserves_scientific_parameters():
    original = {'task_id':'T','task_key':'K','stage':'dft_single_point','status':'failed',
                'task_checksum':'stale','planned_relative_cost':2,'parameters':{'incar_settings':{}}}
    result = restart_failed_task({'tasks':[original]}, 'T', budget_limits=default_budget_rules(), config_version='C')
    assert result['status']=='prepared'
    assert 'task_checksum' not in result['task']
    assert result['task']['parameters']==original['parameters']
    assert result['task']['stage']=='dft_single_point'

from phase_agent.tools.workflows.tool_outcomes import _apply_execution_result
from phase_agent.tools.budget.reserve_action_budget import reserve_action_budget
from phase_agent.tools.policy.validate_tool_action import validate_tool_action


def test_exception_keeps_budget_and_requires_reconciliation():
    action = {'tool': 'generate_branches', 'task_key': 'initial', 'budget': 12}
    old = reserve_action_budget({}, action, config_version='v1')
    new, _ = _apply_execution_result(old, action, {'status': 'failed', 'error': 'partial write possible'}, record_id='a1', formal=True)
    assert new['effective_decisions']['initial']['status'] == 'reconciliation_required'
    assert new['budget_reservations']['initial']['reconciliation_required'] is True
    assert new['budget_reservations']['initial']['status'] == 'reserved'
    assert new['reserved_relative_cost'] == 12
    assert old['effective_decisions']['initial']['status'] == 'reserved'


def test_reviewed_failure_gets_fresh_reservation_with_history():
    state = {'effective_decisions': {'initial': {'status': 'failed', 'verified_no_effect': True}}, 'budget_reservations': {'initial': {'status': 'settled', 'reserved_cost': 12}}, 'reserved_relative_cost': 0}
    action = {'tool': 'generate_branches', 'task_key': 'initial', 'budget': 20}
    result = reserve_action_budget(state, action, config_version='v2')
    assert result['reserved_relative_cost'] == 20
    assert result['budget_reservations']['initial']['config_version'] == 'v2'
    assert result['reservation_history'][0]['reservation']['reserved_cost'] == 12
    assert state['reserved_relative_cost'] == 0
    assert reserve_action_budget(result, action, config_version='v2') == result


def test_unreviewed_failure_not_replaced():
    state = {'effective_decisions': {'initial': {'status': 'reconciliation_required'}}}
    assert reserve_action_budget(state, {'tool': 'generate_branches', 'task_key': 'initial', 'budget': 20}, config_version='v2') == state


def test_mismatch_and_unreconciled_key_rejected_before_dispatch():
    action = {'tool': 'generate_branches', 'task_key': 'initial', 'budget': 0, 'parameters': {'generation_plan': [{'strategy': 'coverage', 'phase': 'O3', 'quota': 120, 'reason': 'coverage'}], 'quotas': {'coverage': 300}, 'total_quota': 300}}
    session = {'status': 'confirmed', 'confirmed_snapshot': {'config': {'agent': {'allowed_tools': ['generate_branches']}, 'budgets': {'total_relative_cost': 100}}, 'config_version': 'v1'}}
    result = validate_tool_action(action, {'effective_decisions': {'initial': {'status': 'reconciliation_required'}}}, session, {'generate_branches': {'handler': lambda **kwargs: None}})
    assert not result['valid']
    assert any(e.startswith('invalid_generation_plan:') for e in result['errors'])
    assert 'duplicate_effective_decision' in result['errors']

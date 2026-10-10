from copy import deepcopy
import pytest
from phase_agent.tools.state.execution_receipts import recovery_report
from phase_agent.tools.dispatch.execute_tool_action import execute_tool_action
from phase_agent.tools.workflows.tool_outcomes import _apply_execution_result
from phase_agent.tools.budget.reserve_action_budget import reserve_action_budget
from phase_agent.runtime.workflow_reply_presentation import format_workflow_reply


def failed_state():
    return {'effective_decisions': {'initial': {'status': 'reserved', 'tool': 'generate_branches'}}, 'action_records': [{'record_id': 'a1', 'status': 'failed', 'final_action': {'task_key': 'initial'}}]}


def test_missing_database_does_not_hide_failed_reserved_action(tmp_path):
    state = failed_state()
    before = deepcopy(state)
    report = recovery_report(tmp_path / 'state.json', state)
    assert report['status'] == 'reconciliation_required'
    assert report['unsettled'][0]['reason'] == 'failed_action_still_reserved'
    assert state == before
    assert not (tmp_path / 'execution_receipts.sqlite').exists()
    reply = format_workflow_reply({'status': 'execution_reconciliation_required', 'recovery_report': report}, tmp_path / 'state.json')
    assert '\u4e0d\u4e00\u81f4' in reply


def test_active_reservation_without_failure_is_not_a_contradiction(tmp_path):
    state = failed_state()
    state['action_records'][0]['status'] = 'awaiting_approval'
    assert recovery_report(tmp_path / 'state.json', state)['status'] == 'clear'


@pytest.mark.parametrize('settled', [False, True])
def test_verified_no_effect_requires_budget_settlement(tmp_path, settled):
    state = {'effective_decisions': {'initial': {'status': 'failed', 'verified_no_effect': True}}, 'budget_reservations': {'initial': {'status': 'settled' if settled else 'reserved'}}}
    assert recovery_report(tmp_path / 'state.json', state)['status'] == ('clear' if settled else 'reconciliation_required')


@pytest.mark.parametrize('partial_output', [False, True])
def test_failed_handler_restart_preserves_output_budget_and_blocks_replay(tmp_path, partial_output):
    calls = []
    def handler(*, action, context):
        calls.append(action)
        if partial_output:
            (tmp_path / 'partial.vasp').write_text('partial output')
        raise RuntimeError('fake interruption')
    action = {'tool': 'generate_branches', 'task_key': 'initial', 'budget': 20}
    context = {'state_path': str(tmp_path / 'state.json'), 'invocation_id': 'a1', 'config_version': 'v1'}
    registry = {'generate_branches': {'handler': handler}}
    state = reserve_action_budget({}, action, config_version='v1')
    execution = execute_tool_action(action, registry=registry, context=context)
    state, _ = _apply_execution_result(state, action, execution, record_id='a1', formal=True)
    state['invocations'] = {'a1': {'status': 'failed', 'execution': execution}}
    assert recovery_report(context['state_path'], state)['status'] == 'reconciliation_required'
    assert state['reserved_relative_cost'] == 20
    assert execute_tool_action(action, registry=registry, context=context)['status'] == 'execution_reconciliation_required'
    assert len(calls) == 1
    assert (tmp_path / 'partial.vasp').exists() == partial_output


def test_returned_failed_receipt_does_not_override_stale_reservation(tmp_path):
    from phase_agent.tools.state.execution_receipts import begin_execution, record_execution_return
    identity = {'invocation_id': 'a1', 'config_version': 'v1', 'action_hash': 'sha256:' + 'a'*64, 'tool': 'generate_branches'}
    path = tmp_path / 'state.json'
    begin_execution(path, identity)
    record_execution_return(path, identity, 'failed')
    state = failed_state()
    state['invocations'] = {'a1': {'status': 'failed', 'execution': {'status': 'failed'}}}
    assert recovery_report(path, state)['unsettled'][0]['reason'] == 'failed_action_still_reserved'


def test_latest_successful_invocation_does_not_inherit_old_failure(tmp_path):
    state = failed_state()
    state['action_records'].append({'record_id': 'a2', 'status': 'completed', 'final_action': {'task_key': 'initial'}})
    state['effective_decisions']['initial']['status'] = 'completed'
    assert recovery_report(tmp_path / 'state.json', state)['status'] == 'clear'

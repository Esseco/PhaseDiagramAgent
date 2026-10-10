from copy import deepcopy
import pytest
from phase_agent.decisions.agent.generation_plan import configured_generation_strategies
from phase_agent.tools.dispatch.execute_tool_action import execute_tool_action
from phase_agent.tools.policy.file_approval import proposal_hash
from phase_agent.runtime.workflow_reply_presentation import _friendly_validation_errors

@pytest.mark.parametrize('phase_scope', [['O3'], {'at_x': {'0': ['O3'], '1': ['O3']}, 'intermediate': ['O3']}])
def test_old_unsynchronized_single_phase_snapshot_is_constrained(phase_scope):
    config = {'system': {'boundary': {'P': phase_scope, 'TM_ratio': {'Cr': 1.0}}}, 'generation_actions': {'enabled': ['coverage', 'composition', 'competing_phase', 'tm_ordering']}}
    original = deepcopy(config)
    assert configured_generation_strategies(config) == ['composition', 'coverage']
    assert config == original


def test_multiphase_multielement_keeps_competition_available():
    config = {'system': {'boundary': {'P': ['O3', 'P3'], 'TM_ratio': {'Fe': .5, 'Mn': .5}}}}
    assert {'competing_phase', 'tm_ordering'} <= set(configured_generation_strategies(config))


@pytest.mark.parametrize('fails', [False, True])
def test_handler_cannot_mutate_approved_proposal_or_receipt_identity(tmp_path, fails):
    action = {'tool': 'generate_branches', 'task_key': 'initial', 'parameters': {'quotas': {'coverage': 300}, 'batch_size': 96}}
    original = deepcopy(action)
    seen = []
    def handler(*, action, context):
        seen.append(deepcopy(action))
        action['parameters']['quotas']['coverage'] = 1
        if fails:
            raise ValueError('fake failure')
        return {'fake': True}
    context = {'state_path': str(tmp_path / 'state.json'), 'invocation_id': 'approved-1', 'config_version': 'v1'}
    registry = {'generate_branches': {'handler': handler}}
    result = execute_tool_action(action, registry=registry, context=context)
    assert result['status'] == ('failed' if fails else 'completed')
    assert seen == [original]
    assert action == original and proposal_hash(action) == proposal_hash(original)
    replay = execute_tool_action(action, registry=registry, context=context)
    assert replay['status'] == 'execution_reconciliation_required'
    assert len(seen) == 1


def test_duplicate_message_does_not_assert_submission():
    message = _friendly_validation_errors(['duplicate_effective_decision'])
    assert '\u5df2\u7ecf\u63d0\u4ea4\u6216\u5b8c\u6210' not in message
    assert '\u5360\u4f4d' in message

def test_reserved_plan_overrides_model_approval_advice():
    from phase_agent.runtime.studio_reply_presentation import format_turn_reply
    state = {'pending_execution_policies': {'p': {'agent_proposal': {'recommended_action': 'generate_branches', 'raw_action': {'tool': 'generate_branches', 'task_key': 'initial', 'parameters': {}}}}}, 'effective_decisions': {'initial': {'status': 'reserved'}}}
    reply = format_turn_reply('\u5efa\u8bae\uff1a\u56de\u590d\u540c\u610f\u6279\u51c6\u3002', state)
    assert '\u5efa\u8bae\uff1a\u56de\u590d\u540c\u610f\u6279\u51c6' not in reply
    assert '\u6682\u4e0d\u53ef\u6279\u51c6' in reply
    assert '\u5df2\u4fdd\u5b58\u65b9\u6848' in reply

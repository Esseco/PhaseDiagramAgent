from copy import deepcopy
from pathlib import Path
import pytest

from tests.integration.test_unified_workflow_bohb import _session, H
from phase_agent.persistence.ledger.phase_data_manager import PhaseDataManager
from phase_agent.runtime.chat_application import RunWorkflowChatHandler
from phase_agent.tools.step_runner.file_protocol import read_json
from phase_agent.tools.state.reconcile_task_results import reconcile_task_results
from phase_agent.tools.budget.reserve_action_budget import reserve_action_budget


def make_handler(tmp_path, model, handlers=None):
    boundary = {'P': ['O3'], 'H': {'O3': [H]}, 'TM_ratio': {'Fe': 1}}
    session = _session(boundary)
    manager = PhaseDataManager(boundary)
    handler = RunWorkflowChatHandler({'manager': manager, 'phase_references': {},
        'run_config': {'state_path': str(tmp_path/'state.json')},
        'state_path': str(tmp_path/'state.json'), 'config_session': session,
        'agent_client': model, 'handlers': handlers or {}}, execution_mode='autonomous')
    return handler


def test_question_is_answered_with_one_call_and_no_proposal(tmp_path):
    calls = []
    def model(payload):
        calls.append(payload)
        return {'kind': 'answer', 'answer': '当前允许O3相；配置中H上限请以已确认值为准。'}
    handler = make_handler(tmp_path, model)
    reply = handler([{'role': 'user', 'content': '解释一下现在的相设置'}], conversation_id='t')
    assert '当前允许O3相' in reply
    assert len(calls) == 1
    assert calls[0]['mode'] == 'autonomous_search'
    assert calls[0]['unified_dialogue'] is True
    assert not read_json(handler.state_path, {}).get('pending_execution_policies')
    assert not read_json(handler.state_path, {}).get('budget_reservations')


def test_propose_question_then_approve_exactly_once(tmp_path):
    calls, executed = [], []
    action = {'tool': 'check_convergence', 'parameters': {}, 'budget': 0,
              'reason': '检查已登记数据', 'expected_purpose': '决定后续搜索'}
    def model(payload):
        calls.append(payload)
        return {'kind': 'answer', 'answer': '这是一次只读收敛检查。'} if len(calls) == 2 else deepcopy(action)
    def tool(**kw):
        executed.append(kw)
        return {'converged': False, 'budget_exhausted': False}
    handler = make_handler(tmp_path, model, {'check_convergence': tool})
    proposed = handler([{'role': 'user', 'content': '分析并建议下一步'}], conversation_id='t')
    assert len(calls) == 1 and not executed
    state = read_json(handler.state_path, {})
    original = deepcopy(state['pending_execution_policies'])
    answer = handler([{'role': 'user', 'content': '这个检查具体有什么作用？'}], conversation_id='t')
    assert '只读收敛检查' in answer
    assert read_json(handler.state_path, {})['pending_execution_policies'] == original
    handler([{'role': 'user', 'content': '同意'}], conversation_id='t')
    assert len(calls) == 2 and len(executed) == 1
    handler([{'role': 'user', 'content': '同意'}], conversation_id='t')
    assert len(executed) == 1 and len(calls) == 3
    # With no stored approval, an affirmative message is semantic input only.
    assert read_json(handler.state_path, {})['pending_execution_policies']


def test_configuration_uses_original_user_request_not_model_invented_patch(tmp_path):
    handler = make_handler(tmp_path, lambda p: {'kind': 'configure', 'answer': '修改远端环境'})
    received = []
    handler.config_revision_factory = lambda state: (
        lambda messages, conversation_id=None: received.append(messages[-1]['content']) or '配置已写入草稿')
    reply = handler([{'role': 'user', 'content': '以后远端DFT都用python环境'}], conversation_id='t')
    assert '配置已写入草稿' in reply
    assert received == ['以后远端DFT都用python环境']


def test_invalid_dialogue_cannot_smuggle_tool_execution(tmp_path):
    executed = []
    handler = make_handler(tmp_path, lambda p: {'kind': 'answer', 'answer': '已完成',
        'tool': 'check_convergence'}, {'check_convergence': lambda **kw: executed.append(kw)})
    reply = handler([{'role': 'user', 'content': '解释一下'}], conversation_id='t')
    assert not executed
    assert not read_json(handler.state_path, {}).get('pending_execution_policies')


def test_memory_is_thread_scoped_and_survives_handler_restart(tmp_path):
    from phase_agent.graphs.dialogue.memory import remember_turn, recent_turns
    state = tmp_path/'state.json'
    remember_turn(state, 'a', '我只研究O3', '已记录')
    assert recent_turns(state, 'a')[0]['user'] == '我只研究O3'
    assert recent_turns(state, 'b') == []
    remember_turn(state, 'a', 'api_key=sk-privatevalue', '已收到')
    assert 'sk-privatevalue' not in str(recent_turns(state, 'a'))


def test_repeated_question_is_a_new_turn_and_rejection_never_executes(tmp_path):
    calls, executed = [], []
    def model(payload):
        calls.append(payload)
        if len(calls) == 1:
            return {'tool': 'check_convergence', 'parameters': {}, 'budget': 0,
                    'reason': '核对数据', 'expected_purpose': '判断下一步'}
        return {'kind': 'answer', 'answer': f'解释{len(calls)}'}
    handler = make_handler(tmp_path, model, {'check_convergence': lambda **kw: executed.append(kw)})
    handler([{'role': 'user', 'content': '建议下一步'}], conversation_id='t')
    for expected in ['解释2', '解释3']:
        assert expected in handler([{'role': 'user', 'content': '再解释一下'}], conversation_id='t')
    handler([{'role': 'user', 'content': '查看方案'}], conversation_id='t')
    handler([{'role': 'user', 'content': '拒绝'}], conversation_id='t')
    assert len(calls) == 3 and not executed
    assert not read_json(handler.state_path, {}).get('pending_execution_policies')


@pytest.mark.parametrize("malformed", [False, True])
def test_return_to_configuration_preserves_pending_proposal(tmp_path, malformed):
    calls, executed, configured = [], [], []
    def model(payload):
        calls.append(payload)
        if len(calls) == 1:
            return {"tool": "check_convergence", "parameters": {}, "budget": 0, "reason": "核对"}
        assert payload["unified_dialogue"]
        assert "dialogue_outcome" in payload["output_contracts"]
        if malformed and len(calls) == 2:
            return {"reply": "返回初始设置"}
        return {"kind": "configure", "answer": "返回初始设置，保留待审批方案"}
    handler = make_handler(tmp_path, model, {"check_convergence": lambda **kw: executed.append(kw)})
    handler.config_revision_factory = lambda state: (
        lambda messages, conversation_id=None: configured.append(messages[-1]["content"]) or "已进入初始设置")
    handler([{"role": "user", "content": "建议下一步"}], conversation_id="a")
    original = deepcopy(read_json(handler.state_path, {})["pending_execution_policies"])
    reply = handler([{"role": "user", "content": "返回初始设置"}], conversation_id="a")
    assert "已进入初始设置" in reply
    assert configured == ["返回初始设置"]
    assert read_json(handler.state_path, {})["pending_execution_policies"] == original
    assert not executed
    assert len(calls) == (3 if malformed else 2)

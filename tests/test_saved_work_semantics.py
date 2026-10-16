import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from phase_agent.graphs.dialogue.saved_control import route_saved_control
from phase_agent.graphs.dialogue.state import TurnRuntime
from tests.test_approval_history_performance import rejected_template_state


def setup(tmp_path, monkeypatch, action, *, unsettled=False):
    client = Mock(return_value=action)
    handler = SimpleNamespace(config_delegate=None, conversation_id='s',
        state_path=tmp_path/'state.json', workflow_kwargs={'agent_client':client},
        _run=Mock(return_value={'status':'awaiting_approval'}))
    monkeypatch.setattr('phase_agent.tools.state.execution_receipts.recovery_report',
        lambda *a: {'unsettled':[{'identity':{}}] if unsettled else []})
    context=TurnRuntime(handler, [], 's')
    return handler, client, context, SimpleNamespace(context=context)


@pytest.mark.parametrize('message', ['接着把刚才那批输入做出来', '沿用刚选的点往下走吧', '可以，按原来的方案接着准备'])
def test_free_form_resume_reuses_selection_once(tmp_path, monkeypatch, message):
    handler, client, context, runtime=setup(tmp_path,monkeypatch,{'kind':'advance'})
    facts=rejected_template_state(); before=deepcopy(facts)
    result=route_saved_control({'facts':facts,'message':message},runtime,[])
    assert result=={'operation':'scientific_workflow'}
    assert client.call_count==1 and facts==before
    payload=client.call_args.args[0]
    assert payload['bounded_control'] and len(json.dumps(payload,ensure_ascii=False))<8000
    assert 'allowed_tools' not in payload and 'state' not in payload
    def run(*args):
        assert handler.workflow_kwargs['frozen_selection_only'] is True
        return {'status':'awaiting_approval'}
    handler._run.side_effect=run
    context.scientific_call()
    assert 'frozen_selection_only' not in handler.workflow_kwargs


def test_question_does_not_resume(tmp_path,monkeypatch):
    h,c,ctx,r=setup(tmp_path,monkeypatch,{'kind':'answer','answer':'原15点方案保留。'})
    assert route_saved_control({'facts':rejected_template_state(),'message':'刚才选点还在吗'},r,[])['reply']=='原15点方案保留。'
    assert ctx.scientific_call is None and c.call_count==1


def test_explicit_revision_uses_existing_versioned_modify_path(tmp_path,monkeypatch):
    h,c,ctx,r=setup(tmp_path,monkeypatch,{'kind':'revise','plan_id':'p'})
    h.review_pending=Mock(return_value={'status':'awaiting_approval','result':{'status':'awaiting_approval'}})
    row={'plan_id':'p','state_version':'v','proposal_hash':'h'}
    result=route_saved_control({'facts':rejected_template_state(),'message':'把这批修改为更少的采点，先给新方案'},r,[row])
    assert result['operation']=='scientific_workflow'
    h.review_pending.assert_not_called()
    assert ctx.scientific_call()['status']=='awaiting_approval'
    assert h.review_pending.call_args.args==('p','modify')
    assert h.review_pending.call_args.kwargs['expected_proposal_hash']=='h'


@pytest.mark.parametrize('decision',['approve','reject'])
def test_semantic_review_is_only_a_request_to_original_validator(tmp_path,monkeypatch,decision):
    h,c,ctx,r=setup(tmp_path,monkeypatch,{'kind':'review','plan_id':'p','decision':decision})
    facts=rejected_template_state()
    result=route_saved_control({'facts':facts,'message':'就按刚刚展示的这份来吧'},r,[{'plan_id':'p'}])
    assert result['operation']=='review' and result['result']['review']['decision']==decision
    assert result['facts'] is facts
    assert ctx.scientific_call is None
    h._run.assert_not_called()


@pytest.mark.parametrize('action', [{'kind':'review','plan_id':'invented','decision':'approve'},
    {'kind':'advance','tool':'select_dft_candidates'}, {'kind':'review','plan_id':'p','decision':'confirm_sensitive'}])
def test_invalid_control_cannot_execute(tmp_path,monkeypatch,action):
    h,c,ctx,r=setup(tmp_path,monkeypatch,action)
    result=route_saved_control({'facts':rejected_template_state(),'message':'那就这么办吧'},r,[{'plan_id':'p'}])
    assert '未取得有效语义判断' in result['reply']
    assert ctx.scientific_call is None and c.call_count==1


def test_recovery_procedure_is_actionable(tmp_path,monkeypatch):
    h,c,ctx,r=setup(tmp_path,monkeypatch,{'kind':'advance'},unsettled=True)
    result=route_saved_control({'facts':{},'message':'先恢复刚才的再继续'},r,[])
    assert '/phase/approval' in result['reply'] and '生成核对处理方案' in result['reply']
    assert ctx.scientific_call is None


def test_exact_approval_has_zero_model_cost(tmp_path,monkeypatch):
    h,c,ctx,r=setup(tmp_path,monkeypatch,{'kind':'advance'})
    assert route_saved_control({'facts':{},'message':'同意'},r,[]) is None
    c.assert_not_called()


def test_failed_semantics_never_retries_or_enters_science(tmp_path,monkeypatch):
    h,c,ctx,r=setup(tmp_path,monkeypatch,{})
    c.side_effect=TimeoutError('test deadline')
    result=route_saved_control({'facts':rejected_template_state(),'message':'按上一批接着准备吧'},r,[])
    assert '未取得有效语义判断' in result['reply']
    assert c.call_count==1 and ctx.scientific_call is None


def test_exact_continue_restores_saved_selection_without_model(tmp_path):
    from phase_agent.graphs.dialogue.nodes import route_request
    model=Mock(side_effect=AssertionError('no model'))
    handler=SimpleNamespace(config_delegate=None,conversation_id=None,state_path=tmp_path/'state.json',
        workflow_kwargs={'agent_client':model},_run=Mock())
    context=TurnRuntime(handler,[],'s')
    result=route_request({'facts':rejected_template_state(),'message':'继续'},SimpleNamespace(context=context))
    assert result['operation']=='scientific_workflow' and callable(context.scientific_call)
    model.assert_not_called()


def test_control_transport_has_deadline_output_limit_and_no_science_contract(monkeypatch):
    from time import monotonic
    from phase_agent.decisions.agent.create_deepseek_client import create_deepseek_client
    captured=[]
    def read(request, timeout, deadline):
        body=json.loads(request.data);captured.append(body)
        assert 0 < deadline-monotonic() <= 12
        assert body['max_tokens']==384 and body['thinking']=={'type':'disabled'}
        payload=json.loads(body['messages'][1]['content'])
        assert 'allowed_tools' not in payload
        return json.dumps({'choices':[{'finish_reason':'stop','message':{'content':'{"kind":"advance"}'}}]}).encode()
    monkeypatch.setattr('phase_agent.runtime.api_deadline.read_response',read)
    client=create_deepseek_client(api_key='fake-test-key',thinking='enabled',total_timeout=120)
    assert client({'bounded_control':True,'semantic_dialogue':True,'mode':'saved_work_control'})['kind']=='advance'
    assert len(captured)==1

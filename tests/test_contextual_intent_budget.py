import json
from unittest.mock import Mock
from run.chat_intent_routing import normalize_chat_intent
from run.conversation_context import conversation_context
from decision_layer.agent.resolve_chat_intent import resolve_chat_intent
from decision_layer.agent.prepare_llm_request import needs_deep_reasoning


def test_one_pass_receives_current_wait_and_recent_answer():
    state={'remote_finetune_jobs':{'j':{'training_handoff':{'stage':'awaiting_validation'}}}}
    client=Mock(return_value={'intent':'continue','direct_request':True,'confidence':.97})
    result=normalize_chat_intent('结果传好了，帮我分析再看看下一轮',state,agent_client=client,
        messages=[{'role':'assistant','content':'请回传独立验证结果'}],enable_context=True,single_pass=True)
    assert result=={'message':'继续'} and client.call_count==1
    context=client.call_args.args[0]['context']['conversation']
    assert context['training_waits'][0]['stage']=='awaiting_validation'
    assert context['recent_conversation'][0]['content']=='请回传独立验证结果'


def test_other_answer_and_clarification_do_not_call_second_model():
    for response in [{'intent':'other','answer':'仍缺独立验证集。'},
                     {'intent':'clarify','clarification':'你指的是清单还是验证结果？'},
                     {'intent':'other'}]:
        client=Mock(return_value=response)
        assert 'reply' in normalize_chat_intent('那个呢',{},agent_client=client,enable_context=True,single_pass=True)
        assert client.call_count==1


def test_context_hard_bound_excludes_raw_evidence():
    state={'active_model_version':'x'*20000,'tasks':[{'status':str(i)} for i in range(1000)],
        'confirmed_config':{'api_key':'secret'},'remote_finetune_jobs':{'j':{
            'directory':'p'*20000,'training_handoff':{'stage':'awaiting_manifest','missing':['x'*20000]}}}}
    context=conversation_context(state,[{'role':'assistant','content':'a'*20000}]*20)
    assert len(json.dumps(context,ensure_ascii=False)) <= 3600
    assert 'secret' not in json.dumps(context)


def test_semantic_read_config_and_feedback_one_pass():
    def response(intent,**fields):
        return lambda _: {'intent':intent,'direct_request':True,'confidence':.98,**fields}
    assert normalize_chat_intent('文件我改完了，核对没问题就往下走',{},agent_client=response('read_config',continue_if_ready=True),
        enable_context=True,single_pass=True)=={'message':'读取配置 JSON 并继续'}
    assert normalize_chat_intent('把方案换成四成员',{},agent_client=response('workflow_feedback'),
        enable_context=True,single_pass=True)=={'message':'把方案换成四成员'}


def test_candidate_review_requires_registered_target_and_keeps_governance():
    context={'candidate_reviews':[{'version':'candidate1','passed':True}]}
    client=lambda _: {'intent':'candidate_review','direct_request':True,'confidence':.98,
        'candidate_model_version':'candidate1','decision':'activate','reason':'独立验证符合标准'}
    result=resolve_chat_intent('验证符合要求，启用这个候选吧',{},agent_client=client,conversation=context)
    assert result['decision']=='activate'
    assert resolve_chat_intent('启用它',{},agent_client=client,conversation={})['intent']=='clarify'
    bad=lambda _: {'intent':'continue','direct_request':True,'confidence':float('nan')}
    assert resolve_chat_intent('接着做',{},agent_client=bad)['intent']=='clarify'
    from execution_layer.local.review_candidate_command import review_candidate_command
    reviewed=review_candidate_command('激活候选 candidate1 原因：批准',{'active_model_version':'old',
        'candidate_models':{'candidate1':{'old_model_version':'old','validation':{'passed':False}}}})
    assert reviewed['state']['active_model_version']=='old'


def test_routing_never_uses_deep_reasoning_for_scientific_words():
    assert not needs_deep_reasoning({'mode':'resolve_chat_intent','user_message':'模型更新预算分配'})

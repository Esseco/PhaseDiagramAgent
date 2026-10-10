import json
from unittest.mock import Mock
from phase_agent.runtime.conversation_context import conversation_context
from phase_agent.decisions.agent.prepare_llm_request import needs_deep_reasoning






def test_context_hard_bound_excludes_raw_evidence():
    state={'active_model_version':'x'*20000,'tasks':[{'status':str(i)} for i in range(1000)],
        'confirmed_config':{'api_key':'secret'},'remote_finetune_jobs':{'j':{
            'directory':'p'*20000,'training_handoff':{'stage':'awaiting_manifest','missing':['x'*20000]}}}}
    context=conversation_context(state,[{'role':'assistant','content':'a'*20000}]*20)
    assert len(json.dumps(context,ensure_ascii=False)) <= 3600
    assert 'secret' not in json.dumps(context)






def test_routing_never_uses_deep_reasoning_for_scientific_words():
    assert not needs_deep_reasoning({'mode':'configuration_dialogue','user_message':'模型更新预算分配'})

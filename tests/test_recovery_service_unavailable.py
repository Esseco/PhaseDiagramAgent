from types import SimpleNamespace
import pytest
from phase_agent.graphs.dialogue.nodes import decide


@pytest.mark.parametrize('message',['回收完毕','回收完成。','回传完成','结果已回传','已回传'])
def test_recovery_notification_defers_science_without_semantic_call(message,monkeypatch):
    from phase_agent.graphs.dialogue import support,decision
    monkeypatch.setattr(support,'fresh_turn_context',lambda *a:{})
    monkeypatch.setattr(decision,'decide_from_saved_facts',lambda *a,**kw:pytest.fail('Unnecessary semantic call'))
    handler=SimpleNamespace(workflow_kwargs={},turn_context={},_run=lambda *a,**kw:pytest.fail('Must defer; no execution'))
    runtime=SimpleNamespace(context=SimpleNamespace(handler=handler,messages=[{'role':'user','content':message}],conversation_id='t'))
    result=decide({'facts':{},'message':message},runtime)
    assert result['operation']=='scientific_workflow' and callable(runtime.context.scientific_call)


def test_503_is_specific_and_not_retried(monkeypatch):
    from urllib.error import HTTPError
    import urllib.request
    from phase_agent.decisions.agent.create_deepseek_client import create_deepseek_client,DeepSeekResponseError
    calls=[]
    def fail(*a,**kw):calls.append(1);raise HTTPError('https://api.deepseek.com',503,'unavailable',{},None)
    monkeypatch.setattr(urllib.request,'urlopen',fail)
    with pytest.raises(DeepSeekResponseError,match='模型服务暂时不可用') as error:create_deepseek_client(api_key='offline')({'state':{}})
    assert len(calls)==1 and error.value.code=='service_unavailable'
    assert error.value.llm_usage['unknown_usage_calls']==1

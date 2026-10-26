import json
from copy import deepcopy
import pytest
from phase_agent.analysis.state.decision_evidence import candidate_evidence, transport_copy
from phase_agent.runtime.chat_application import RunWorkflowChatHandler


def test_projection_preserves_all_metrics_constraints_and_approval_parameters():
    payload = {"state": {"available_branches": [{"candidate_id": str(i), "energy": i, "model_version": "m1", "composition": {"Fe": .5}, "details": {"structure": {"sites": [1]*100}, "ehull": .1}} for i in range(120)], "human_constraints": ["no relax"]}, "current_proposal": {"parameters": {"structure": {"sites": [1]}}}}
    before=deepcopy(payload);projected=transport_copy(payload)
    assert len(projected['state']['available_branches'])==120
    assert projected['state']['available_branches'][-1]['candidate_id']=='119'
    assert projected['state']['available_branches'][0]['details']=={'ehull':.1}
    assert projected['current_proposal']==payload['current_proposal']
    projected['state']['human_constraints'].append('other')
    assert payload==before


def test_heavy_values_are_never_copied():
    class CannotCopy:
        def __deepcopy__(self, memo):raise AssertionError('atom payload copied')
    assert candidate_evidence([{'candidate_id':'a','structure':CannotCopy()}])==[{'candidate_id':'a'}]


@pytest.mark.parametrize('message',['查看进度','查看方案','微调文件夹在哪','回传到哪里','查看上次回复'])
def test_query_bypasses_recovery_model_and_workflow(tmp_path, monkeypatch, message):
    from phase_agent.tools.workflows import recover_saved_results
    def forbidden(*args,**kwargs):pytest.fail('Read-only query dispatched scientific work')
    monkeypatch.setattr(recover_saved_results,'recover_saved_results',forbidden)
    path=tmp_path/'state.json';path.write_text(json.dumps({'tasks':[], 'pending_execution_policies':{}, 'relax_mc_batches':{'b':{'status':'prepared','upload_folder':'input','expected_results_archive':'output/results.zip'}}}))
    before=path.read_bytes()
    handler=RunWorkflowChatHandler({'state_path':str(path),'agent_client':forbidden},workflow=forbidden,history_prompt=True)
    assert handler([{'role':'user','content':message}],conversation_id='t')
    assert path.read_bytes()==before


def test_unreadable_state_cannot_dispatch(tmp_path):
    path=tmp_path/'state.json';path.write_text('{bad')
    def forbidden(*args,**kwargs):pytest.fail('State error dispatched workflow')
    handler=RunWorkflowChatHandler({'state_path':str(path),'agent_client':forbidden},workflow=forbidden)
    with pytest.raises(json.JSONDecodeError):handler([{'role':'user','content':'查看进度'}])


def test_missing_state_cannot_dispatch(tmp_path):
    from phase_agent.graphs.dialogue.errors import OpenWebUIRequestError
    def forbidden(*args, **kwargs):pytest.fail('Missing state dispatched workflow')
    (tmp_path/'langgraph_lifecycle.sqlite').touch()
    handler=RunWorkflowChatHandler({'state_path':str(tmp_path/'missing.json'),'agent_client':forbidden},workflow=forbidden)
    with pytest.raises(OpenWebUIRequestError, match='未分析或生成任务'):
        handler([{'role':'user','content':'继续'}])


def test_action_is_not_a_read_only_query(tmp_path):
    from phase_agent.runtime.read_only_queries import read_only_reply
    from types import SimpleNamespace
    handler=SimpleNamespace(config_delegate=None,state_path=tmp_path/'state.json')
    for message in ['继续','重新生成上传目录','上传完成，继续分析']:
        assert read_only_reply(handler,message,{},'t') is None
    assert '当前没有可批准' in read_only_reply(handler, '同意', {}, 't')


def test_cancelled_inputs_are_not_offered(tmp_path):
    from phase_agent.runtime.read_only_queries import read_only_reply
    from types import SimpleNamespace
    handler=SimpleNamespace(config_delegate=None,state_path=tmp_path/'state.json')
    reply=read_only_reply(handler,'回传到哪里',{'relax_mc_batches':{'b':{'status':'cancelled','upload_folder':'bad'}},'slurm_batches':[{'batch_id':'x','status':'cancelled','handoff':{'folder':'bad'}}]})
    assert 'bad' not in reply

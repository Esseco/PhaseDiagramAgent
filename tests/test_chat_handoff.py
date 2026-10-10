from types import SimpleNamespace
from langchain_core.messages import HumanMessage
from phase_agent.graphs.studio_chat_graph import build_studio_graph
from phase_agent.graphs.dialogue.nodes import route_request
from phase_agent.runtime.studio_reply_presentation import format_turn_reply


def test_only_one_node_emits_reply(tmp_path):
    graph=build_studio_graph(lambda *_:'proposal question',receipt_path=tmp_path/'gateway.json')
    updates=list(graph.stream({'messages':[HumanMessage(content='status',id='m1')]},{'configurable':{'thread_id':'t1'}},stream_mode='updates'))
    emitted=[message for update in updates for body in update.values() if isinstance(body,dict) for message in body.get('messages',[])]
    assert len(emitted)==1 and emitted[0].content=='proposal question'
    assert not any(body.get('reply')=='proposal question' for update in updates for body in update.values() if isinstance(body,dict))


def test_yes_without_saved_plan_goes_to_semantic_model():
    runtime=SimpleNamespace(context=SimpleNamespace(handler=SimpleNamespace(config_delegate=None),conversation_id='t'))
    assert route_request({'message':'同意','facts':{}},runtime)['operation']=='decide'
    assert route_request({'message':'同意','facts':{'pending_execution_policies':{'p':{}}}},runtime)['operation']=='review'


def test_generated_results_offer_relax_proposal_not_generic_continue():
    reply=format_turn_reply('已生成 72 个 branch、156 个初始结构。',{'generation_history':[{'registered_ids':['s1']}], 'tasks':[]})
    assert '是否现在' in reply and 'Relax' in reply
    assert '回复‘继续’' not in reply

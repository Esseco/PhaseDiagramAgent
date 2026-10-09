from unittest.mock import Mock
import pytest
from decision_layer.agent.hybrid_proposal import create_hybrid_client,choose_harness


def test_routine_never_constructs_deep_client():
    legacy=Mock(return_value={'intent':'status','_llm_usage':{'calls':1}})
    factory=Mock()
    client=create_hybrid_client(legacy,factory)
    result=client({'mode':'resolve_chat_intent','analysis_harness':'deepagents'})
    assert result['_proposal_harness']=='legacy'
    factory.assert_not_called()


def test_complex_lazy_factory_usage_and_no_double_spend():
    legacy=Mock(return_value={'tool':'pause_search'})
    deep=Mock(return_value={'tool':'check_convergence','_llm_usage':{'calls':2,'input_tokens':100}})
    factory=Mock(return_value=deep);client=create_hybrid_client(legacy,factory)
    payload={'mode':'autonomous_search','decision_context':{'round_budget_evidence':{
        'required':True,'training_reports':[{'report_id':'a'},{'report_id':'b'}]}}}
    result=client(payload);client(payload)
    assert result['_proposal_harness']=='deepagents' and result['_llm_usage']['calls']==2
    assert result['_analysis_route']['elapsed_seconds']>=0
    assert factory.call_count==1
    legacy.assert_not_called()
    deep.side_effect=RuntimeError('failed')
    with pytest.raises(RuntimeError):client(payload)
    legacy.assert_not_called()


def test_single_report_and_normal_dft_remain_lightweight():
    assert choose_harness({'mode':'autonomous_search','decision_context':{'qbc_candidates':[{}]*100}})[0]=='legacy'
    assert choose_harness({'mode':'autonomous_search','decision_context':{
        'round_budget_evidence':{'training_reports':[{}]}}})[0]=='legacy'


def test_explicit_complex_request_only_in_proposal_mode():
    assert choose_harness({'mode':'autonomous_search','analysis_harness':'deepagents'})[0]=='deepagents'
    assert choose_harness({'mode':'configuration_json_review','analysis_harness':'deepagents'})[0]=='legacy'

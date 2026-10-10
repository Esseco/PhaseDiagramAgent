from copy import deepcopy
import pytest
from phase_agent.decisions.agent.dialogue_contract import dialogue_errors, dialogue_result
from tests.test_unified_react_dialogue import make_handler


def test_evidence_metadata_is_preserved_without_execution_authority():
    action = {'kind':'answer','answer':'Explain failure','evidence_refs':['last_turn.error']}
    state = {'pending_execution_policies': {'saved': {'status':'awaiting_approval'}}}
    original = deepcopy(state)
    assert dialogue_errors(action, True) == []
    result = dialogue_result(action, state)
    assert result['evidence_refs'] == ['last_turn.error']
    assert result['steps_executed'] == 0 and result['submitted'] is False
    assert state == original


@pytest.mark.parametrize('refs', ['state.json',[{}],[1],[''],['x']*33])
def test_evidence_refs_remain_typed(refs):
    assert dialogue_errors({'kind':'answer','answer':'Explanation','evidence_refs':refs},True)


def test_executable_fields_are_still_rejected():
    assert dialogue_errors({'kind':'answer','answer':'Explanation','evidence_refs':[], 'tool':'generate_branches'},True)


def test_question_with_evidence_uses_production_dialogue_without_science(tmp_path):
    calls=[]
    def model(payload):
        calls.append(payload)
        return {'kind':'answer','answer':'The previous proposal failed validation; nothing was executed.','evidence_refs':['last_turn.error']}
    handler=make_handler(tmp_path, model)
    handler._run=lambda *a,**kw: pytest.fail('Read-only explanation entered science')
    answer=handler([{'role':'user','content':'\u4ec0\u4e48\u95ee\u9898'}],conversation_id='evidence')
    assert 'nothing was executed' in answer
    assert len(calls)==1

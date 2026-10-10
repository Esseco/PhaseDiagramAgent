from phase_agent.runtime.studio_reply_presentation import format_turn_reply
from phase_agent.runtime.workflow_reply_presentation import format_workflow_reply


def test_completed_result_precedes_stage_and_advice():
    text=format_turn_reply('\u5df2\u751f\u6210 96 branch\n\u4e0b\u4e00\u6b65\uff1aRelax requires approval',{})
    lines=text.splitlines()
    assert lines[0].startswith('\u7ed3\u679c\uff1a') and '96' in lines[0]
    assert lines[1].startswith('\u5f53\u524d\u72b6\u6001\uff1a')
    assert lines[2].startswith('\u4e0b\u4e00\u6b65\uff1a') and 'approval' in lines[2]


def test_proposed_work_does_not_become_completed_result():
    text=format_turn_reply('\u5efa\u8bae\uff1agenerate 96 branches\nrequires approval',{})
    assert '\u7ed3\u679c\uff1a' not in text
    assert 'requires approval' in text


def test_failure_keeps_actual_cause_and_no_execution():
    text=format_turn_reply('\u672c\u8f6e\u672a\u6267\u884c\uff1amissing reference\n\u4e0b\u4e00\u6b65\uff1aprovide O3.vasp',{})
    assert text.splitlines()[0].startswith('\u7ed3\u679c\uff1a')
    assert 'missing reference' in text and 'O3.vasp' in text


def test_ordinary_answer_is_not_wrapped_by_workflow():
    assert format_workflow_reply({'status':'answered','answer':'A concise explanation'},'state.json')=='A concise explanation'

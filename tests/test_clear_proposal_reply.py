from copy import deepcopy
from phase_agent.runtime.workflow_reply_presentation import format_workflow_reply
from phase_agent.runtime.chat_state_presentation import format_epoch_reply


def proposal():
    params = {'total_quota': 300, 'batch_size': 100, 'initial_states_per_branch': 3, 'max_det_H': 12, 'generation_plan': [{'strategy': strategy, 'phase': 'O3', 'quota': count, 'reason': 'model rationale'} for strategy, count in [('coverage',180), ('composition',80), ('periodic_extension',40)]], 'generation_cost_preview': {'relax_mc_cost_upper': 6046, 'serial_seconds_upper': None}}
    return {'status': 'awaiting_approval', 'agent_proposal': {'recommended_action': 'generate_branches', 'expected_purpose': 'coverage', 'estimated_cost': {'estimated_total_cost': 0}, 'action_parameters': params, 'raw_action': {'tool':'generate_branches','parameters':params}}}


def test_brief_branch_approval_distinguishes_counts_and_scope():
    result = proposal()
    original = deepcopy(result)
    reply = format_workflow_reply(result, None, verbose=False)
    for text in ['300', '180', '80', '40', '100', '基础覆盖', 'Na 含量补充', '超胞补充', '去重', '实际数量', '不运行 Relax、MC 或 DFT', '同意', '拒绝']:
        assert text in reply
    assert '预计相对成本：0' not in reply
    assert '后续Relax+MC成本粗估' not in reply
    assert result == original


def test_brief_preserves_allocation_constraints():
    result = proposal()
    row = result['agent_proposal']['action_parameters']['generation_plan'][0]
    row.update(na_min=0.25, na_max=0.75, max_det_H=6)
    reply = format_workflow_reply(result, None, verbose=False)
    assert 'Na/O₂ 0.25–0.75' in reply
    assert 'det(H) ≤ 6' in reply and 'det(H) ≤ 12' in reply


def test_details_retain_future_cost_and_scientific_reasons():
    reply = format_workflow_reply(proposal(), None, verbose=True)
    assert 'model rationale' in reply
    assert '后续Relax+MC成本粗估上界' in reply
    assert '非本次派发授权' in reply


def test_epoch_header_removes_only_identical_stage():
    state = {'pending_execution_policies': {'p': {'agent_proposal': {'recommended_action': 'generate_branches', 'raw_action': {'tool':'generate_branches'}}}}}
    text = format_epoch_reply('当前状态：branch 生成方案待确认。\n下一步：核对方案。', state)
    assert text.count('branch 生成方案待确认') == 1
    assert '下一步：核对方案。' in text
    other = format_epoch_reply('当前状态：执行记录与占位不一致。', state)
    assert '执行记录与占位不一致' in other

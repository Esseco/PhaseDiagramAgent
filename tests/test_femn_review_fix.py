import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from phase_agent.configuration.defaults.default_layered_search_config import default_layered_search_config
from phase_agent.configuration.session.project_config_json import create_project_config_json
from phase_agent.configuration.session.load_editable_config_json import load_editable_config_json
from phase_agent.analysis.state.project_authority import configuration_changes
from phase_agent.runtime.editable_config_baseline import editable_comparison_baseline
from phase_agent.runtime.concise_dialogue import concise_body
from phase_agent.decisions.agent.branch_pool_review import pool_context, review_errors
from phase_agent.decisions.agent.propose_tool_action import propose_agent_tool_action
from phase_agent.tools.policy.region_sampling import policy
from phase_agent.tools.workflows.region_review import write_proposal
from phase_agent.runtime.review_presentation import review_card


def test_missing_optional_threshold_is_not_an_unapplied_edit(tmp_path):
    config = default_layered_search_config()
    config['convergence'].pop('final_force_rmse_tolerance')
    path = tmp_path / 'config.json'
    create_project_config_json(path, config)
    document = json.loads(path.read_text().split('\n', 1)[1])
    document['overrides'].setdefault('convergence', {})['final_force_rmse_tolerance'] = None
    path.write_text(json.dumps(document))
    baseline = editable_comparison_baseline(config, path, tmp_path)
    editable = load_editable_config_json(path, config)
    assert "convergence.final_force_rmse_tolerance" not in configuration_changes(baseline, editable)
    document['overrides']['convergence']['final_force_rmse_tolerance'] = 0.07
    path.write_text(json.dumps(document))
    editable = load_editable_config_json(path, config)
    assert 'convergence.final_force_rmse_tolerance' in configuration_changes(baseline, editable)


def test_repeated_next_step_retains_scope_and_query_boundary():
    text = concise_body('结果：待审方案。\n下一步：查看该方案后回复“同意”“拒绝”或提出修改；本次查询不会批准。\n下一步：回复‘同意’或提出修改。确认范围：只登记，不计算。', {'pending_execution_policies': {'x': {}}})
    assert text.count('下一步：') == 1
    assert '只登记，不计算' in text and '本次查询不会批准' in text and '拒绝' in text


def review_action():
    return {'tool': 'generate_branches', 'task_key': 'review-existing', 'target_ids': [], 'budget': 0,
            'reason': '先复用已有池', 'expected_purpose': '审阅后停止',
            'parameters': {'sampling_mode': 'region_adaptive', 'regions': [], 'defer_reason': '先搜索已有池'},
            'branch_pool_review': {'choice': 'reuse_existing', 'sufficiency_for_next_search': '足够启动下一批，未证明收敛',
                                   'rationale': '本轮新增尚未回收', 'alternative_reason': '先回收再判断补充收益',
                                   'missing_evidence': ['回收结果'], 'revisit_conditions': ['回收后'], 'evidence_refs': ['ledger:current']}}


def test_project_agent_can_defer_without_fixed_quota(tmp_path):
    captured = []
    def client(payload):
        captured.append(payload)
        return review_action()
    state = {'branch_count_policy': policy(), 'generation_history': [{'new_branch_ids': ['b'], 'registered_ids': ['s']}],
             'active_model_version': 'm1', 'decision_context': {'evidence_catalog': {'references': {'ledger:current': {}}}}}
    result = propose_agent_tool_action(state, agent_client=client, allowed_tools=['generate_branches'], config={})
    assert result['decision_source'] == 'llm_agent', result.get('fallback_reason')
    assert result['parameters']['regions'] == []
    assert 'total_quota' not in result['parameters']
    context = captured[0]['decision_context']
    assert context['branch_pool_review_required']
    assert context['branch_count_policy']['count_is_result']
    assert '每轮选择100' not in json.dumps(context, ensure_ascii=False)
    state['workspace_layout'] = {'workspace_root': str(tmp_path)}
    result['parameters']['region_preview'] = {'existing_branch_count': 1}
    payload = json.loads(__import__('pathlib').Path(write_proposal({'raw_action': result}, state)['json']).read_text())
    assert payload['本次新增branch数'] == 0
    assert payload['已有branch是否需要补充']['建议'] == '先复用已有branch，暂不补充'
    card = review_card('p', {'agent_proposal': {'raw_action': result}})
    assert '暂缓' in card['title'] and '另行' in card['approval_boundary']


@pytest.mark.parametrize('bad', ['missing', 'choice', 'rationale', 'refs'])
def test_missing_or_contradictory_review_rejected(bad):
    action = review_action()
    if bad == 'missing':
        action.pop('branch_pool_review')
    elif bad == 'choice':
        action['branch_pool_review']['choice'] = 'supplement'
    elif bad == 'rationale':
        action['branch_pool_review']['rationale'] = ''
    else:
        action['branch_pool_review']['evidence_refs'] = 'made up'
    assert review_errors(action, {'branch_pool_review_required': True})


def test_supplement_is_a_model_choice_not_a_count_rule():
    action = review_action()
    action['parameters']['regions'] = [{'region_id': 'r'}]
    action['branch_pool_review']['choice'] = 'supplement'
    assert review_errors(action, {'branch_pool_review_required': True}) == []


def test_pool_count_is_latest_registered_batch_and_search_is_not_invented():
    facts = pool_context({'generation_history': [{'new_branch_ids': ['old']}, {'new_branch_ids': ['a', 'b'], 'registered_ids': ['s']}], 'tasks': []})
    assert facts['registered_new_branch_count'] == 2
    assert facts['related_task_states'] == {}


def test_deferred_header_and_next_step_do_not_ask_to_generate(tmp_path):
    from phase_agent.runtime.chat_state_presentation import format_epoch_reply
    from phase_agent.runtime.status_presentation import format_progress
    action = review_action()
    state = {"active_model_version": "m", "upload_layout": {"model_rounds": {"m": 2}},
             "pending_execution_policies": {"p": {"agent_proposal": {"recommended_action": "generate_branches", "raw_action": action}}}}
    text = format_epoch_reply(format_progress(state) + "\n下一步：回复同意。确认范围：仅确认暂缓新增，后续另审。", state)
    assert "已有branch复用方案待确认" in text
    assert "暂缓新增branch" in text
    assert text.count("下一步：") == 1

from copy import deepcopy
import pytest
from phase_agent.configuration.defaults.default_layered_search_config import default_layered_search_config
from phase_agent.configuration.session.project_config_json import create_project_config_json, expand_project_config
from phase_agent.configuration.session.load_editable_config_json import _strip_jsonc_comments
from phase_agent.configuration.session.materialize_layered_h import materialize_layered_h
from phase_agent.configuration.session.create_config_draft import create_config_draft
from phase_agent.configuration.session.apply_config_revision import apply_config_revision
from phase_agent.configuration.session.confirm_config_snapshot import confirm_config_snapshot
from phase_agent.configuration.schema.validate_configuration_space import validate_configuration_space
from phase_agent.decisions.agent.generation_plan import configured_generation_strategies
from phase_agent.decisions.agent.proposal_validation import proposal_errors
import json


def test_new_project_has_no_invented_scientific_boundary(tmp_path, monkeypatch):
    path = tmp_path / 'project.json'
    assert create_project_config_json(path)
    doc = json.loads(_strip_jsonc_comments(path.read_text(encoding='utf-8')))
    config = expand_project_config(doc, source=path)
    assert config['system']['boundary'] == {'P': [], 'H': {}, 'TM_ratio': {}}
    assert config['system']['configuration_space']['roles']['T'] is None
    called = []
    monkeypatch.setattr('phase_agent.configuration.session.materialize_layered_h.enumerate_layered_oxide_supercells', lambda *a, **kw: called.append(1))
    with pytest.raises(ValueError, match='system.boundary.P'):
        materialize_layered_h(config)
    assert called == []
    result = confirm_config_snapshot(create_config_draft(config), user_confirmed=True)
    assert result['status'] == 'draft'
    assert result['confirmed_snapshot'] is None


def test_explicit_project_boundary_and_freedom_are_preserved(tmp_path):
    config = default_layered_search_config(boundary={'P': ['O3'], 'H': {}, 'TM_ratio': {'Cr': 1.0}})
    path = tmp_path / 'project.json'
    create_project_config_json(path, config)
    loaded = expand_project_config(json.loads(_strip_jsonc_comments(path.read_text(encoding='utf-8'))), source=path)
    assert loaded['system']['boundary']['P'] == ['O3']
    assert loaded['system']['boundary']['TM_ratio'] == {'Cr': 1.0}


@pytest.mark.parametrize('ratio,fixed,expected', [({'Cr': 1}, False, False), ({'Fe': .5, 'Mn': .5}, True, False), ({'Fe': .5, 'Mn': .5}, False, True)])
def test_tm_freedom_is_independent_of_component_count(ratio, fixed, expected):
    config = default_layered_search_config(boundary={'P': ['O3'], 'H': {}, 'TM_ratio': ratio})
    if fixed:
        config['system']['configuration_space'].update(roles={'P':'branch','H':'branch','x':'branch','T':'fixed','N':'internal'},fixed_T_source='phase_reference')
    original = deepcopy(config)
    enabled = configured_generation_strategies(config)
    assert ('tm_ordering' in enabled) == expected
    assert 'competing_phase' not in enabled
    assert config == original


def test_natural_language_patch_contract_disables_fixed_tm_without_changing_ratio():
    config = default_layered_search_config(boundary={'P': ['O3','P3'], 'H': {}, 'TM_ratio': {'Fe': .5, 'Mn': .5}})
    # LLM supplies the semantic patch; deterministic code enforces its meaning.
    session = apply_config_revision(create_config_draft(config), {'system.configuration_space.roles.T':'fixed', 'system.configuration_space.fixed_T_source':'phase_reference'})
    c = session['config']
    assert c['system']['boundary']['TM_ratio'] == {'Fe': .5, 'Mn': .5}
    enabled = configured_generation_strategies(c)
    assert 'tm_ordering' not in enabled and 'competing_phase' in enabled
    assert validate_configuration_space(c['system'])['valid']
    action = {'tool':'generate_branches','task_key':'test','target_ids':[],'budget':0,'reason':'test','expected_purpose':'test','parameters':{'total_quota':1,'quotas':{'tm_ordering':1},'generation_plan':[{'strategy':'tm_ordering','quota':1,'phase':'O3','reason':'test'}]}}
    errors = proposal_errors(action, {'allowed_tools':['generate_branches'],'decision_context':{'enabled_generation_strategies':enabled}})
    assert any('disabled strategies' in e for e in errors)
    assert action['parameters']['quotas']['tm_ordering'] == 1


def test_fixed_variables_constrain_strategies_and_validate_values():
    config = default_layered_search_config(boundary={'P':['O3','P3'],'H':{'O3':[[[1,0,0],[0,1,0],[0,0,1]]]},'TM_ratio':{'Fe':1,'Mn':1}})
    space = config['system']['configuration_space']
    space['roles'].update(P='fixed',H='fixed',x='fixed',T='fixed')
    space.update(fixed_T_source='phase_reference',fixed_values={'P':'O3','H':[[1,0,0],[0,1,0],[0,0,1]],'x':'1/2'})
    assert configured_generation_strategies(config) == ['coverage']
    assert validate_configuration_space(config['system'])['valid']
    space['fixed_values'].update(P='O1',x=2)
    errors = validate_configuration_space(config['system'])['errors']
    assert any('fixed_values.P' in e for e in errors)
    assert any('fixed_values.x' in e for e in errors)

def test_analysis_adapter_preserves_semantic_scope():
    from phase_agent.decisions.agent.deepagents_proposal import proposal_material
    context = {'scientific_scope':{'boundary':{'P':{'at_x':{'0':['O3'],'1':['O3']},'intermediate':['O3']},'TM_ratio':{'Fe':.5,'Mn':.5}},'configuration_space':{'roles':{'T':'fixed'},'fixed_T_source':'phase_reference'},'instruction':'respect scope'},'enabled_generation_strategies':['coverage'],'generation_strategy_constraint':'respect fixed variables'}
    out = proposal_material({'decision_context':context})['decision_context']
    assert out['scientific_scope'] == context['scientific_scope']
    assert out['enabled_generation_strategies'] == ['coverage']


def test_long_template_also_requires_explicit_boundary(tmp_path):
    from phase_agent.configuration.session.create_editable_config_json import create_editable_config_json
    path = tmp_path / 'draft.json'
    create_editable_config_json(path, default_layered_search_config())
    config = json.loads(_strip_jsonc_comments(path.read_text(encoding='utf-8')))['config']
    assert config['system']['boundary'] == {'P': [],'H': {},'TM_ratio': {}}
    assert config['system']['configuration_space']['roles']['T'] is None

def test_single_tm_resolves_unset_occupancy_without_an_extra_question(tmp_path):
    path = tmp_path / 'project.json'
    create_project_config_json(path)
    doc = json.loads(_strip_jsonc_comments(path.read_text(encoding='utf-8')))
    doc['config']['system']['boundary'].update(P=['O3'], TM_ratio={'Cr':1.0})
    c = expand_project_config(doc, source=path)
    assert c['system']['configuration_space']['roles']['T'] == 'fixed'
    assert c['system']['configuration_space']['fixed_T_source'] == 'phase_reference'
    assert c['system']['branch_schema']['fields'] == ['P','H','x']
    assert 'tm_ordering' not in configured_generation_strategies(c)


def test_scope_summary_displays_fixed_binary_occupancy():
    from phase_agent.configuration.session.summarize_config_for_agent import format_scientific_scope
    c = default_layered_search_config(boundary={'P':['O3'],'H':{},'TM_ratio':{'Fe':1,'Mn':1}})
    c['system']['configuration_space']['roles']['T'] = 'fixed'
    c['system']['configuration_space']['fixed_T_source'] = 'phase_reference'
    summary = format_scientific_scope(c)
    assert 'Fe' in summary and 'Mn' in summary and 'phase_reference' in summary
    assert 'tm_ordering' not in summary and 'competing_phase' not in summary

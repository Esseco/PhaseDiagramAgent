"""Presentation binds a saved plan, never approves or executes it."""
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from phase_agent.runtime.read_only_queries import read_only_reply
from phase_agent.graphs.dialogue.support import _load_binding, review_presented_proposal


@pytest.mark.parametrize('query', ['现在什么状态', '查看方案'])
@pytest.mark.parametrize('entry', ['fast', 'inspection'])
def test_display_then_first_approval(tmp_path, monkeypatch, query, entry):
    state = {'pending_execution_policies': {'p': {'agent_proposal': {
        'recommended_action': 'generate_branches'}}}}
    original = deepcopy(state)
    (tmp_path / 'state.json').write_text(json.dumps(state), encoding='utf-8')
    monkeypatch.setattr('phase_agent.runtime.plan_queries.pending_plan_reply', lambda s: 'specific saved plan')
    monkeypatch.setattr('phase_agent.graphs.dialogue.commands.format_status_reply', lambda s: 'status')
    monkeypatch.setattr('phase_agent.runtime.chat_approval_rules.is_sensitive_proposal', lambda p: False)
    monkeypatch.setattr('phase_agent.runtime.workflow_reply_presentation.format_workflow_reply', lambda *a: 'executed')
    handler = SimpleNamespace(state_path=tmp_path / 'state.json', conversation_id=None,
                              config_delegate=None, review_pending=Mock(return_value={'result': {}}))
    if entry == 'fast':
        reply = read_only_reply(handler, query, state, 'new-session')
    else:
        from phase_agent.graphs.dialogue.maintenance.inspection import handle
        reply = handle(handler, query, state, [], 'new-session')
    assert 'specific saved plan' in reply
    assert handler.conversation_id == 'new-session'
    binding = _load_binding(handler)
    assert binding['plan_id'] == 'p'
    handler.review_pending.assert_not_called()
    assert state == original
    assert review_presented_proposal(handler, state, '同意') == 'executed'
    handler.review_pending.assert_called_once_with('p', 'approve', expected_state_version=binding['state_version'],
                                                expected_proposal_hash=binding['proposal_hash'], comment='同意')


@pytest.mark.parametrize('count', [0, 2])
def test_ambiguous_or_no_plan_never_binds(tmp_path, monkeypatch, count):
    state = {'pending_execution_policies': {str(i): {'agent_proposal': {
        'recommended_action': 'generate_branches'}} for i in range(count)}}
    monkeypatch.setattr('phase_agent.graphs.dialogue.commands.format_status_reply', lambda s: 'status')
    monkeypatch.setattr('phase_agent.runtime.plan_queries.pending_plan_reply', lambda s: 'choose plan')
    handler = SimpleNamespace(state_path=tmp_path / 'state.json', conversation_id=None, config_delegate=None)
    read_only_reply(handler, '现在什么状态', state, 'new')
    assert _load_binding(handler) is None


@pytest.mark.parametrize('project', ['O3-1ele-NaCoO2', 'O3-1ele-NaCrO2', 'O3-2ele-NaFeMn'])
@pytest.mark.real_project
def test_existing_project_copy_binds_without_mutating_project(tmp_path, project, monkeypatch):
    # The existing renderer writes a display artifact to configured project paths.
    # Stub only that artifact writer; retain real formatting and approval binding.
    monkeypatch.setattr('phase_agent.tools.workflows.dft_selection_artifacts.proposal_artifacts', lambda *a, **k: None)
    monkeypatch.setattr('phase_agent.tools.workflows.generation_artifacts.proposal_artifacts', lambda *a, **k: None)
    source = Path('E:/0-PhaseDiagram') / project / 'workflow_state/state.json'
    if not source.is_file():
        pytest.skip('local project unavailable')
    before = source.read_bytes()
    state = json.loads(before)
    from phase_agent.runtime.review_requests import pending_reviews
    rows = pending_reviews(state)
    assert len(rows) <= 1
    copy_path = tmp_path / 'state.json'
    copy_path.write_bytes(before)
    handler = SimpleNamespace(state_path=copy_path, conversation_id=None, config_delegate=None)
    read_only_reply(handler, '现在什么状态', state, 'fresh-project-session')
    binding = _load_binding(handler)
    if rows:
        assert binding['plan_id'] == rows[0]['plan_id']
        assert binding['proposal_hash'] == rows[0]['proposal_hash']
    else:
        assert binding is None
    assert source.read_bytes() == before
    assert copy_path.read_bytes() == before


@pytest.mark.parametrize('guard', ['sensitive', 'changed'])
def test_existing_approval_guards_remain(tmp_path, monkeypatch, guard):
    state = {'pending_execution_policies': {'p': {'agent_proposal': {
        'recommended_action': 'generate_branches'}}}}
    (tmp_path / 'state.json').write_text(json.dumps(state), encoding='utf-8')
    monkeypatch.setattr('phase_agent.runtime.plan_queries.pending_plan_reply', lambda s: 'saved plan')
    monkeypatch.setattr('phase_agent.graphs.dialogue.commands.format_status_reply', lambda s: 'status')
    monkeypatch.setattr('phase_agent.runtime.chat_approval_rules.is_sensitive_proposal', lambda p: guard == 'sensitive')
    handler = SimpleNamespace(state_path=tmp_path / 'state.json', conversation_id=None,
                              config_delegate=None, review_pending=Mock(side_effect=ValueError('changed')))
    read_only_reply(handler, 'status', state, 'a')
    other = SimpleNamespace(state_path=handler.state_path, conversation_id='b')
    assert _load_binding(other) is None
    reply = review_presented_proposal(handler, state, '同意')
    if guard == 'sensitive':
        assert '/phase/approval' in reply
        handler.review_pending.assert_not_called()
    else:
        assert '未执行任务' in reply
        handler.review_pending.assert_called_once()
        assert _load_binding(handler) is None

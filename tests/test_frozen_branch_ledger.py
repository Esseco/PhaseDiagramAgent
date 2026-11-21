from types import SimpleNamespace

import pytest

from phase_agent.tools.workflows import react_proposal


def test_new_frozen_branch_review_reads_live_ledger(monkeypatch):
    manager = object()
    calls = []
    monkeypatch.setattr(react_proposal, 'agent_state_summary',
                        lambda state, *, manager=None, _approval_only=False: calls.append((manager, _approval_only)) or {})
    monkeypatch.setattr('phase_agent.tools.workflows.mc_coverage_choice.resolve', lambda frame: None)
    monkeypatch.setattr('phase_agent.tools.workflows.initial_tool_proposal.select_initial_proposal',
                        lambda **kwargs: {'status': 'checked'})
    frame = {'current': {'pending_execution_policies': {}},
             'context': {'saved_input_only': True, 'manager': manager},
             'agent_client': SimpleNamespace(_codex_structured=True, _frozen_generation_draft=True),
             'invocation_id': 'i', 'human_feedback': None, 'config': {}, 'registry': {}}
    assert react_proposal.prepare_react_proposal(frame)['status'] == 'checked'
    assert calls == [(manager, False)]
    frame['context'].pop('manager')
    with pytest.raises(ValueError, match='frozen_branch_requires_live_ledger'):
        react_proposal.prepare_react_proposal(frame)


def test_existing_approved_proposal_retains_bounded_summary(monkeypatch):
    calls = []
    monkeypatch.setattr(react_proposal, 'agent_state_summary',
                        lambda state, *, manager=None, _approval_only=False: calls.append((manager, _approval_only)) or {})
    monkeypatch.setattr('phase_agent.tools.workflows.mc_coverage_choice.resolve', lambda frame: None)
    monkeypatch.setattr(react_proposal, 'validate_tool_action', lambda *a: {'valid': True})
    frame = {'current': {'pending_execution_policies': {'i': {'agent_proposal': {'raw_action': {'tool': 'generate_branches'}}, 'record_id': 'r'}}},
             'context': {'saved_input_only': True, 'manager': object()},
             'agent_client': SimpleNamespace(_codex_structured=True, _frozen_generation_draft=True),
             'invocation_id': 'i', 'human_feedback': {'decision': 'reject'}, 'config': {}, 'registry': {}, 'session': {}}
    result = react_proposal.prepare_react_proposal(frame)
    assert result['_graph_prepared'] is True
    assert calls == [(None, True)]

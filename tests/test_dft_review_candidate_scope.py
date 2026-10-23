from copy import deepcopy

import pytest

from phase_agent.configuration.defaults.default_budget_rules import default_budget_rules
from phase_agent.configuration.defaults.default_dft_decision_config import (
    default_dft_decision_config,
)
from phase_agent.tools.workflows.create_dft_selection_handler import (
    _state_candidates,
    create_dft_selection_handler,
)


def candidate(name):
    return {
        "candidate_id": name,
        "structure_id": name,
        "branch_id": "branch-" + name,
        "atom_count": 10,
        "predicted_Ehull": 0.02,
        "qbc": {"status": "completed"},
    }


@pytest.mark.parametrize("decision", ["DEFER", "REJECT"])
def test_nonexecuting_decisions_remain_available_for_approved_validation(decision, monkeypatch):
    state = {
        "qbc_candidates": [candidate("selected"), candidate("later"), candidate("other")],
        "tasks": [],
        "pending_tasks": [],
    }
    action = {
        "target_ids": ["selected"],
        "tool": "select_dft_candidates",
        "parameters": {
            "decisions": [
                {"candidate_id": "selected", "action": "DFT_SINGLE_POINT", "reason": "verify"},
                {"candidate_id": "later", "action": decision, "reason": "later"},
            ]
        },
    }
    original = deepcopy(state)
    available = _state_candidates(state, None, action)
    assert {r["candidate_id"] for r in available} == {"selected", "later"}
    monkeypatch.setattr(
        "phase_agent.tools.workflows.dft_selection_artifacts.preparation_artifacts",
        lambda *a, **k: {},
    )
    config = {
        "qbc": default_dft_decision_config(),
        "budgets": default_budget_rules(),
        "mlip": {"version": "m"},
        "dft": {"parameters": {}},
    }
    result = create_dft_selection_handler()(
        action=action,
        context={"event_state": state, "effective_config": config, "config_version": "v"},
    )
    assert result["validation"]["valid"]
    assert [r["structure_id"] for r in result["tasks"]] == ["selected"]
    assert state == original


def test_selected_candidate_outside_target_is_not_added_to_execution_scope():
    action = {
        "target_ids": ["selected"],
        "parameters": {
            "decisions": [
                {"candidate_id": "outside", "action": "DFT_SINGLE_POINT", "reason": "verify"}
            ]
        },
    }
    rows = _state_candidates(
        {"qbc_candidates": [candidate("selected"), candidate("outside")]}, None, action
    )
    assert {r["candidate_id"] for r in rows} == {"selected"}


def test_unknown_deferred_candidate_is_not_fabricated():
    action = {
        "target_ids": ["selected"],
        "parameters": {
            "decisions": [{"candidate_id": "unknown", "action": "DEFER", "reason": "later"}]
        },
    }
    rows = _state_candidates({"qbc_candidates": [candidate("selected")]}, None, action)
    assert {r["candidate_id"] for r in rows} == {"selected"}

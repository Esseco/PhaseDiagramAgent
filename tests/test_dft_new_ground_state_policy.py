from phase_agent.analysis.phase.phase_dft_candidates import bind_ground_state_frames
from phase_agent.configuration.defaults.default_dft_decision_config import (
    default_dft_decision_config,
)
from phase_agent.tools.workflows.validate_dft_agent_decisions import validate_dft_agent_decisions
from tests.test_dft_ground_state_binding import fixture


def test_new_strict_rule_blocks_gs_deferral_but_legacy_contract_is_preserved(tmp_path, monkeypatch):
    from phase_agent.evidence import dft_extrapolation

    monkeypatch.setattr(
        dft_extrapolation, "candidate_extrapolation", lambda *a: {"status": "normal"}
    )
    state, candidate, _ = fixture(tmp_path)
    state["qbc_candidates"] = bind_ground_state_frames(state, [candidate])
    metrics = [
        {"candidate_id": "other", "branch_id": "b2", "atom_count": 10},
        {"candidate_id": "s1", "branch_id": "b1", "atom_count": 4},
    ]
    proposal = {
        "decisions": [
            {"candidate_id": "other", "action": "DFT_SINGLE_POINT", "reason": "coverage"},
            {"candidate_id": "s1", "action": "DEFER", "reason": "later"},
        ],
        "source": "codex",
    }
    kwargs = {
        "config": default_dft_decision_config(),
        "config_version": "c1",
        "remaining_budget": 2000,
    }
    old = validate_dft_agent_decisions(proposal, metrics, state, **kwargs)
    new = validate_dft_agent_decisions(
        proposal, metrics, state, require_all_ground_states=True, **kwargs
    )
    assert "ground_state_selection_required:s1" not in old["errors"]
    assert "ground_state_selection_required:s1" in new["errors"]


def test_new_preview_rejects_reasoned_deferral_of_gs(tmp_path, monkeypatch):
    from phase_agent.configuration.defaults.default_layered_search_config import (
        default_layered_search_config,
    )
    from phase_agent.evidence import dft_extrapolation
    from phase_agent.tools.workflows.preview_dft_inputs import preview_dft_inputs

    monkeypatch.setattr(
        dft_extrapolation, "candidate_extrapolation", lambda *a: {"status": "normal"}
    )
    state, candidate, _ = fixture(tmp_path)
    other = {
        "candidate_id": "other",
        "structure_id": "other",
        "branch_id": "b2",
        "atom_count": 10,
        "phase": "O3",
    }
    state["qbc_candidates"] = bind_ground_state_frames(state, [candidate, other])
    action = {
        "tool": "select_dft_candidates",
        "parameters": {
            "decisions": [
                {"candidate_id": "other", "action": "DFT_SINGLE_POINT", "reason": "coverage"},
                {"candidate_id": "s1", "action": "DEFER", "reason": "later"},
            ]
        },
    }
    result, error = preview_dft_inputs(action, state, default_layered_search_config())
    assert result is None
    assert "ground_state_selection_required:s1" in error


def test_execution_revalidates_new_marker_but_preserves_legacy(monkeypatch):
    import importlib

    module = importlib.import_module("phase_agent.tools.workflows.create_dft_selection_handler")
    calls = []

    def validate(*args, **kwargs):
        calls.append(kwargs["require_all_ground_states"])
        return {"valid": False, "rejected": [], "errors": ["synthetic stop"]}

    monkeypatch.setattr(module, "validate_dft_agent_decisions", validate)
    handler = module.create_dft_selection_handler(candidates_provider=lambda *a: [])
    context = {
        "event_state": {},
        "effective_config": {"qbc": {}, "dft": {}, "budgets": {}},
        "config_version": "v",
    }
    for params in ({"dft_selection_preview": {"ground_state_policy": "all_eligible_gs_v1"}}, {}):
        handler(action={"parameters": params}, context=context)
    assert calls == [True, False]


def test_direct_selection_keeps_gs_exemption_under_new_policy(tmp_path, monkeypatch):
    from phase_agent.evidence import dft_extrapolation
    from phase_agent.tools.workflows import validate_dft_agent_decisions as validator

    monkeypatch.setattr(
        dft_extrapolation, "candidate_extrapolation", lambda *a: {"status": "normal"}
    )
    monkeypatch.setattr(validator, "_is_direct_recovery_selection", lambda *a: True)
    state, candidate, _ = fixture(tmp_path)
    state["qbc_candidates"] = bind_ground_state_frames(state, [candidate])
    metrics = [{"candidate_id": "other", "branch_id": "b2", "atom_count": 10}]
    proposal = {
        "decisions": [{"candidate_id": "other", "action": "DFT_SINGLE_POINT", "reason": "DIRECT"}],
        "source": "codex",
    }
    result = validator.validate_dft_agent_decisions(
        proposal,
        metrics,
        state,
        config=default_dft_decision_config(),
        config_version="v",
        remaining_budget=2000,
        require_all_ground_states=True,
    )
    assert result["valid"]

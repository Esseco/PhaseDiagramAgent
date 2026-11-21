import json

from phase_agent.agents.strategic.scientific_decisions import scientific_decision_errors
from phase_agent.runtime.local_agent_control import _scientific_validation_context
from tests.test_three_scientific_decisions import review


def test_native_team_proposal_uses_current_post_dft_context(tmp_path, monkeypatch):
    import phase_agent.evidence.post_dft_assessment as module

    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"confirmed_config": {"label": "c"}}), encoding="utf8")
    seen = []
    def assessment(state, config):
        seen.append((state, config))
        return {"status": "evaluated"}
    monkeypatch.setattr(module, "post_dft_assessment", assessment)
    action = {"_decision_team": {"team_id": "t"}, "tool": "select_dft_candidates",
              "parameters": {"decisions": [{"candidate_id": "x", "action": "DFT_SINGLE_POINT"}]},
              "scientific_decision": {**review("round_direction"),
                  "selection_purposes": [{"candidate_id": "x", "purpose": "coverage"}]}}
    brief = {"config_version": "c1", "model_version": "m1"}
    context = {"other": "retained"}
    assert "scientific_decision_point_mismatch" in scientific_decision_errors(action, brief, context)
    actual = _scientific_validation_context(action, context, state_path)
    assert actual == {"other": "retained", "post_dft_assessment": {"status": "evaluated"}}
    assert not scientific_decision_errors(action, brief, actual)
    assert seen[0][1] == {"label": "c"}
    assert context == {"other": "retained"}


def test_non_team_proposal_keeps_original_context(tmp_path, monkeypatch):
    import phase_agent.evidence.post_dft_assessment as module
    monkeypatch.setattr(module, "post_dft_assessment", lambda *args: (_ for _ in ()).throw(AssertionError("unexpected read")))
    context = {"post_dft_assessment": None}
    assert _scientific_validation_context({"tool": "select_dft_candidates"}, context, tmp_path / "missing") is context

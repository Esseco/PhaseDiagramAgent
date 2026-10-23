from copy import deepcopy
from phase_agent.tools.workflows.dft_coverage_review import coverage_evidence
from phase_agent.tools.workflows.prepare_dft_proposal import prepare_dft_proposal
from tests.test_dft_DH_coverage import candidate


def test_numerical_H_residuals_do_not_create_physical_questions():
    pool = [
        candidate("noise", None, h=8.2e-11),
        candidate("boundary", None, h=1e-8),
        candidate("real", None, h=1e-5),
        candidate("D", "independent-D", h=0),
    ]
    assert [g["candidate_ids"] for g in coverage_evidence(pool)["groups"]] == [["real"], ["D"]]
    assert pool[0]["dft_sampling_evidence"]["H_single_max"] == 8.2e-11


def test_single_repair_receives_all_defects_and_complete_original_decisions(monkeypatch):
    import phase_agent.tools.workflows.prepare_dft_proposal as module

    state = {"qbc_candidates": [candidate("exact-A", "A"), candidate("exact-B", "B")]}
    action = {
        "tool": "select_dft_candidates",
        "parameters": {"decisions": [{"candidate_id": "typo-A", "action": "DFT_SINGLE_POINT"}]},
    }
    before = deepcopy(action)
    calls = []

    def revise(proposal, opinion, **kwargs):
        feedback = kwargs["state"]["decision_context"]["dft_validation_feedback"]
        calls.append(feedback)
        assert "unknown_candidate:typo-A" in feedback["error"]
        assert "exact-A" in feedback["error"] and "exact-B" in feedback["error"]
        assert feedback["prior_decisions"] == action["parameters"]["decisions"]
        assert feedback["allowed_candidate_ids"] == ["exact-A", "exact-B"]
        fixed = deepcopy(action)
        fixed["parameters"]["decisions"] = [
            {"candidate_id": "exact-A", "action": "DFT_SINGLE_POINT"},
            {
                "candidate_id": "exact-B",
                "action": "DEFER",
                "reason": "conditional group requires prior validation",
            },
        ]
        return {"revision_status": "revised", "action": fixed}

    monkeypatch.setattr(module, "attach_dft_preview", lambda a, *args: (deepcopy(a), None))
    result = prepare_dft_proposal(action, state, {}, {}, agent_client=lambda p: None, revise=revise)
    assert result.status == "ready" and len(calls) == 1 and action == before


def test_repair_cannot_lose_previously_reviewed_group(monkeypatch):
    import phase_agent.tools.workflows.prepare_dft_proposal as module

    state = {"qbc_candidates": [candidate("a", "A"), candidate("b", "B")]}
    original = {
        "tool": "select_dft_candidates",
        "parameters": {"decisions": [{"candidate_id": "a", "action": "DFT_SINGLE_POINT"}]},
    }
    calls = []

    def revise(*args, **kwargs):
        calls.append(1)
        return {
            "revision_status": "revised",
            "action": {
                "tool": "select_dft_candidates",
                "parameters": {"decisions": [{"candidate_id": "b", "action": "DFT_SINGLE_POINT"}]},
            },
        }

    monkeypatch.setattr(module, "attach_dft_preview", lambda a, *args: (deepcopy(a), None))
    result = prepare_dft_proposal(
        original, state, {}, {}, agent_client=lambda p: None, revise=revise
    )
    assert result.status == "rejected" and result.action == original and len(calls) == 1
    assert "A:a" in result.error

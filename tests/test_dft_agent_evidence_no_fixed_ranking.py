from phase_agent.configuration.defaults.default_layered_search_config import (
    default_layered_search_config,
)
from phase_agent.tools.workflows.dft_proposal_evidence import refresh_dft_evidence


def test_formal_evidence_does_not_invoke_fixed_ranker(monkeypatch):
    import phase_agent.agents.reliability.selection.recommend_post_mc_dft as legacy

    def forbidden(*args, **kwargs):
        raise AssertionError("fixed weighted recommendation must not run")

    monkeypatch.setattr(legacy, "recommend_post_mc_dft", forbidden)
    cfg = default_layered_search_config()
    candidates = [
        {"candidate_id": f"s{i}", "structure_id": f"s{i}", "atom_count": 12} for i in range(20)
    ]
    state = {"qbc_candidates": candidates, "tasks": [], "active_model_version": "m"}
    context = {}
    refresh_dft_evidence(state, context, cfg)
    reference = context["decision_context"]["dft_sampling_recommendation"]
    assert reference["candidate_ids"] == []
    assert reference["summary"]["status"] == "agent_decision_required"
    assert state["dft_audit_candidate_ids"]
    before = list(state["dft_audit_candidate_ids"])
    refresh_dft_evidence(state, context, cfg)
    assert state["dft_audit_candidate_ids"] == before


def test_codex_rule_audit_does_not_call_weighted_selector(monkeypatch):
    import phase_agent.agents.reliability.selection.select_dft_candidates as legacy
    from phase_agent.runtime.codex.context import build_codex_rule_audit

    def forbidden(*args, **kwargs):
        raise AssertionError("fixed weighted selector must not run")

    monkeypatch.setattr(legacy, "select_dft_candidates", forbidden)
    state = {"qbc_candidates": [{"candidate_id": "s"}], "dft_audit_candidate_ids": []}
    action = {
        "tool": "select_dft_candidates",
        "parameters": {
            "decisions": [{"candidate_id": "s", "action": "DFT_SINGLE_POINT", "reason": "GS"}]
        },
    }
    audit = build_codex_rule_audit(action, state, {})
    assert audit["policy_ranked_candidate_ids"] == []
    assert audit["policy_overlap_count"] is None

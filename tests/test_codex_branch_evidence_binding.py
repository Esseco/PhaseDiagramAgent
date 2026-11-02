from types import SimpleNamespace

from phase_agent.agents.exploration.branch_pool_review import pool_context
from phase_agent.evidence.decision_evidence_catalog import decision_evidence_catalog
from phase_agent.tools.state.state_manager import agent_state_summary


def test_live_proposal_uses_same_ledger_backed_pool_as_codex_context():
    state = {
        "active_model_version": "m",
        "branch_candidates": [{"branch_id": "b1", "P": "O3", "x": "1/2"}],
        "current_state_snapshot": {"decision_context": {}},
    }
    branches = {"b1": {"P": "O3", "x": "1/2", "det_H": 4, "T": ["Ni"]}}
    manager = SimpleNamespace(data={"branches": branches})
    expected = pool_context(state, branches)
    summary = agent_state_summary(state, manager=manager)
    context = summary["decision_context"]
    assert context["branch_pool_facts"] == expected
    expected_refs = decision_evidence_catalog({"branch_pool_facts": expected})["references"]
    assert set(expected_refs) <= set(context["evidence_catalog"]["references"])
    assert agent_state_summary(state)["decision_context"]["branch_pool_facts"] != expected

from copy import deepcopy

from phase_agent.analysis.state.build_decision_context import build_decision_context
from phase_agent.analysis.phase.phase_dft_sampling import dft_sampling_reference
from phase_agent.decisions.agent.deepagents_proposal import proposal_material
from tests.test_phase_quality import sample


def test_current_audit_constraints_and_validation_feedback_reach_project_agent():
    state = sample()
    state["dft_audit_candidate_ids"] = ["audit1", "audit2"]
    before = deepcopy(state)
    context = build_decision_context(state)
    assert state == before
    context["dft_validation_feedback"] = {
        "error": "independent_dft_audit_required",
        "instruction": "Reconsider without execution",
    }
    context["dft_selection_policy"] = {
        "max_tasks_per_round": 100,
        "single_point_cost_per_round": None,
        "reference_framework": dft_sampling_reference(),
    }
    material = proposal_material(
        {"decision_context": context, "allowed_tools": ["select_dft_candidates"]}
    )
    transmitted = material["decision_context"]
    assert transmitted["independent_audit_candidate_ids"] == ["audit1", "audit2"]
    assert transmitted["dft_validation_feedback"]["error"] == "independent_dft_audit_required"
    assert transmitted["dft_selection_policy"]["max_tasks_per_round"] == 100
    assert transmitted["dft_selection_policy"]["single_point_cost_per_round"] is None
    assert transmitted["dft_selection_policy"]["reference_framework"] == dft_sampling_reference()

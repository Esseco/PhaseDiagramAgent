"""Scientific DFT guards used by the production selection handler."""
import pytest
from phase_agent.configuration.defaults.default_dft_decision_config import default_dft_decision_config
from phase_agent.tools.workflows.validate_dft_agent_decisions import validate_dft_agent_decisions


def validate(decisions, *, budget=2000, duplicate=False):
    metric = {"candidate_id": "near", "branch_id": "B1", "f_std_max": 1.0,
              "f_std_p95": 0.8, "energy_std": 0.2, "predicted_Ehull": 0.01,
              "estimated_cost": 1}
    if duplicate:
        metric["duplicate_of"] = "original"
    return validate_dft_agent_decisions(
        {"decisions": decisions, "source": "llm_agent"}, [metric], {},
        config=default_dft_decision_config(), config_version="c1", remaining_budget=budget)


@pytest.mark.parametrize("decision", [
    {"candidate_id": "unknown", "action": "DFT_RELAX"},
    {"candidate_id": "near", "action": "DFT_RELAX", "dft_parameters": {"encut": 300}},
])
def test_unknown_candidate_and_parameter_override_are_rejected(decision):
    result = validate([decision])
    assert not result["valid"]
    assert result["accepted"] == []


def test_duplicate_cannot_become_a_new_dft_task():
    result = validate([{"candidate_id": "near", "action": "DFT_SINGLE_POINT"}], duplicate=True)
    assert result["accepted"] == []
    assert result["rejected"][0]["reason"] == "duplicate_safety_rule"


def test_exhausted_budget_cannot_reserve_a_task():
    result = validate([{"candidate_id": "near", "action": "DFT_SINGLE_POINT"}], budget=0)
    assert result["accepted"] == []
    assert result["rejected"][0]["reason"] == "dft_budget"

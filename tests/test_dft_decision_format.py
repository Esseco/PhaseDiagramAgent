import pytest
from phase_agent.analysis.cost.estimate_proposal_cost import estimate_proposal_cost
from phase_agent.tools.workflows.validate_dft_agent_decisions import validate_dft_agent_decisions


@pytest.mark.parametrize("decisions", ["DFT_RELAX", {"s": "DFT_RELAX"}, ["DFT_RELAX"]])
def test_malformed_decisions_never_crash(decisions):
    cost = estimate_proposal_cost({"tool": "select_dft_candidates",
        "parameters": {"decisions": decisions}}, {})
    assert cost["estimated_total_cost"] is None
    result = validate_dft_agent_decisions({"decisions": decisions}, [], {},
        config={}, config_version="v", remaining_budget=10)
    assert not result["valid"] and result["accepted"] == []

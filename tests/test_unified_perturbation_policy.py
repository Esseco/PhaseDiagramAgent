import math
import pytest
from phase_agent.evidence.extrapolation_policy import assess_policy
from phase_agent.policies.execution_policy import build_agent_proposal

@pytest.mark.parametrize("updates", [None, 0, 1, 2, 20])
@pytest.mark.parametrize("fraction,status", [(0.099,"clear"),(0.1,"sampling_required"),(0.101,"sampling_required"),(0.2,"sampling_required"),(None,"unknown"),(math.nan,"unknown")])
def test_new_policy(updates,fraction,status):
    result=assess_policy(updates,fraction,0,0,False)
    assert result=={"status":status,"route":"perturbation" if status=="sampling_required" else None}

def test_legacy_policy_retains_frozen_route():
    assert assess_policy(2,0.15,0,0,False,legacy=True)["route"]=="extrapolation"
    assert assess_policy(2,0.2,0,0,False,legacy=True)["route"]=="perturbation"

@pytest.mark.parametrize("operation,mode", [("sample_direct_extrapolation","direct_extrapolation"),("prepare_perturbation_recovery","perturbation_recovery_inputs")])
def test_new_proposal_cannot_choose_legacy_entry(operation,mode):
    with pytest.raises(ValueError,match="new_recovery_requires_parent_first"):
        build_agent_proposal({"tool":"prepare_local_batch_files","operation":operation,"parameters":{"mode":mode}}, {})

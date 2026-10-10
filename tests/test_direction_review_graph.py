import pytest
from phase_agent.graphs.direction_review_graph import direction_checkpoint


def test_restart_resumes_same_proposal_and_routes(tmp_path):
    path = tmp_path / "state.json"
    proposal = {"direction": "activate", "candidate_model_version": "new", "reason": "improved"}
    paused = direction_checkpoint(path, proposal)
    assert "decision" not in paused
    assert direction_checkpoint(path, proposal) == paused
    result = direction_checkpoint(path, proposal, "approve")
    assert result["status"] == "approved"
    assert direction_checkpoint(path, proposal, "approve") == result
    with pytest.raises(ValueError, match="already_recorded"):
        direction_checkpoint(path, proposal, "reject")


def test_changed_evidence_requires_new_approval(tmp_path):
    path = tmp_path / "state.json"
    proposal = {"direction": "supplement_dft", "fingerprint": "one"}
    assert direction_checkpoint(path, proposal, "approve")["status"] == "approved"
    changed = {**proposal, "fingerprint": "two"}
    assert "decision" not in direction_checkpoint(path, changed)
    assert direction_checkpoint(path, changed, "reject")["status"] == "rejected"

from phase_agent.tools.state.approved_direction import approved_direction, direction_hash, direction_action_errors


def test_action_is_bound_to_approved_direction_and_evidence():
    proposal = {"direction": "supplement_dft", "training_fingerprint": "one"}
    state = {"active_model_version": "old", "remote_finetune_jobs": {"j": {
        "original_model_version": "old", "training_handoff": {
            "direction_status": "approved", "direction_proposal": proposal}}}}
    assert approved_direction(state) == proposal
    action = {"tool": "select_dft_candidates", "_approved_direction_hash": direction_hash(proposal)}
    assert direction_action_errors(action, state) == []
    assert "action_outside_approved_direction" in direction_action_errors({**action, "tool": "update_mlip"}, state)
    proposal["training_fingerprint"] = "two"
    assert "action_not_bound_to_current_direction" in direction_action_errors(action, state)
    state["remote_finetune_jobs"]["j"]["activated"] = True
    assert "approved_direction_changed" in direction_action_errors(action, state)

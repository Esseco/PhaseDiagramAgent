from phase_agent.tools.state.training_pending_validation import training_pending_validation


def test_only_current_unactivated_return_blocks_training():
    state = {"active_model_version": "base", "remote_finetune_jobs": {"j": {
        "status": "results_received", "original_model_version": "base", "activated": False}}}
    assert training_pending_validation(state)
    state["remote_finetune_jobs"]["j"]["activated"] = True
    assert not training_pending_validation(state)

from phase_agent.tools.policy.branch_conditions import branch_entry_errors, branch_facts


def test_returned_training_blocks_stale_training_action():
    state = {"active_model_version": "old", "remote_finetune_jobs": {
        "j": {"status": "results_received", "original_model_version": "old", "activated": False}}}
    assert "training_already_received_review_required" in branch_entry_errors({"tool": "update_mlip"}, state)
    assert branch_entry_errors({"tool": "select_dft_candidates"}, state) == []
    assert branch_facts(state)["training_review_pending"] is True
    state["remote_finetune_jobs"]["j"]["activated"] = True
    assert branch_entry_errors({"tool": "update_mlip"}, state) == []


def test_refresh_blocks_new_search_but_allows_exact_refresh_action():
    state = {"model_refresh": {"status": "waiting_results"}}
    assert branch_entry_errors({"tool": "generate_branches"}, state)
    assert branch_entry_errors({"tool": "prepare_local_batch_files", "parameters": {"mode": "model_refresh_inputs"}}, state) == []
    state["model_refresh"]["status"] = "completed"
    assert branch_entry_errors({"tool": "generate_branches"}, state) == []

from phase_agent.runtime.chat_state_presentation import _current_imported_mc_identity_review


def _state(registered):
    return {
        "active_model_version": "model-new",
        "branch_candidates": [{"branch_id": value} for value in registered],
        "relax_mc_batches": {
            "batch": {
                "model_version": "model-new",
                "status": "completed",
                "import_checksum": "sha256",
                "branch_ids": ["B-1", "B-2"],
                "result_summary": {"task_count": 2, "failed_task_ids": []},
            }
        },
        "tasks": [
            {"relax_mc_bundle_id": "batch", "task_id": "T-1", "status": "completed"},
            {"relax_mc_bundle_id": "batch", "task_id": "T-2", "status": "completed"},
        ],
    }


def test_completed_batch_with_registered_branches_is_not_reported_as_misaligned():
    assert _current_imported_mc_identity_review(_state(["B-1", "B-2"])) is None


def test_completed_batch_still_reports_missing_branch_identity():
    message = _current_imported_mc_identity_review(_state(["B-1"]))
    assert message and "仍有 1 个 batch branch 未登记" in message

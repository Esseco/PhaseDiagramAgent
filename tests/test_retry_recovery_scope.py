from phase_agent.tools.state.dft_recovery_decision import update_dft_recovery_question
from phase_agent.evidence.dft_round_status import dft_round_scope


def test_registered_retry_is_not_missing_return_of_original_attempt():
    original = dict(task_id="old", stage="dft_single_point", status="failed", upload_operation_id="op", model_version="m")
    retry = dict(original, task_id="retry", retry_of="old", retry_index=1, status="pending")
    state = update_dft_recovery_question({"tasks": [original, retry]})
    assert not state.get("pending_dft_recovery_question")
    assert state["tasks"] == [original, retry]
    assert dft_round_scope(original) != dft_round_scope(retry)


def test_partial_returns_within_retry_attempt_still_require_decision():
    a = dict(task_id="retry-a", stage="dft_single_point", status="completed", upload_operation_id="op", model_version="m", retry_of="old-a", retry_index=1)
    b = dict(a, task_id="retry-b", retry_of="old-b", status="submitted")
    state = update_dft_recovery_question({"tasks": [a, b]})
    assert state["pending_dft_recovery_question"]["pending_task_ids"] == ["retry-b"]

import pytest
from phase_agent.graphs.wait_contract import wait_boundary


@pytest.mark.parametrize("status,kind", [
    ("awaiting_approval", "approval"), ("awaiting_manual_submission", "results"),
    ("tasks_in_progress", "results"), ("execution_reconciliation_required", "reconciliation"),
    ("approval_reconciliation_required", "reconciliation"), ("awaiting_dft_recovery_decision", "input")])
def test_wait_kind_never_grants_authority(status, kind):
    boundary = wait_boundary(status)
    assert boundary["kind"] == kind
    assert boundary["resume_authorizes_execution"] is False
    assert boundary["resume_requirement"]


@pytest.mark.parametrize("status", ["completed", "failed", "converged", "not_configured", None])
def test_terminal_or_error_is_not_a_results_wait(status):
    assert wait_boundary(status) is None

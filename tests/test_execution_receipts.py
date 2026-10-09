from concurrent.futures import ThreadPoolExecutor
import pytest
from execution_layer.state.execution_receipts import begin_execution, record_execution_return
from execution_layer.dispatch.execute_tool_action import execute_tool_action
from orchestration.approval_graph import durable_approval


IDENTITY = {"invocation_id": "a", "config_version": "c", "action_hash": "h", "tool": "generate_branches"}


def test_concurrent_claim_has_only_one_winner(tmp_path):
    path = tmp_path / "state.json"
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: begin_execution(path, IDENTITY), range(2)))
    assert sum(result["allowed"] for result in results) == 1


def test_returned_result_does_not_authorize_replay(tmp_path):
    path = tmp_path / "state.json"
    assert begin_execution(path, IDENTITY)["allowed"]
    record_execution_return(path, IDENTITY, "completed")
    assert begin_execution(path, IDENTITY) == {"allowed": False, "reason": "execution_unsettled", "phase": "returned"}
    assert begin_execution(path, {**IDENTITY, "config_version": "new"})["reason"] == "identity_changed"
    assert begin_execution(tmp_path / "other.json", IDENTITY)["allowed"]


def test_tool_failure_and_business_loss_never_replay(tmp_path):
    calls = []
    def tool(**kwargs):
        calls.append(1)
        raise RuntimeError("may have written a file")
    context = {"state_path": str(tmp_path / "state.json"), "config_version": "c", "invocation_id": "a"}
    action = {"tool": "generate_branches", "task_key": "key"}
    registry = {"generate_branches": {"handler": tool}}
    assert execute_tool_action(action, registry=registry, context=context)["status"] == "failed"
    assert execute_tool_action(action, registry=registry, context=context)["status"] == "execution_reconciliation_required"
    assert len(calls) == 1


@pytest.mark.parametrize("patch", [{"revision": True}, {"revision": -1}, {"revision": "1"}, {"invocation_id": " "}, {"extra": 1}])
def test_invalid_approval_contract_does_not_create_checkpoint(tmp_path, patch):
    envelope = {"invocation_id": "a", "config_version": "c", "proposal_hash": "h", "revision": 0}
    with pytest.raises(ValueError):
        durable_approval(tmp_path / "state.json", {**envelope, **patch})
    assert not (tmp_path / "langgraph_approvals.sqlite").exists()


def test_invalid_execution_contract_does_not_create_receipt(tmp_path):
    with pytest.raises(ValueError):
        begin_execution(tmp_path / "state.json", {**IDENTITY, "config_version": None})
    assert not (tmp_path / "execution_receipts.sqlite").exists()


def test_process_exit_after_claim_is_not_replayed(tmp_path):
    calls = []
    def tool(**kwargs):
        calls.append(1)
        raise SystemExit("simulated termination during tool")
    action = {"tool": "generate_branches", "task_key": "key"}
    context = {"state_path": str(tmp_path / "state.json"), "config_version": "c", "invocation_id": "a"}
    registry = {"generate_branches": {"handler": tool}}
    with pytest.raises(SystemExit):
        execute_tool_action(action, registry=registry, context=context)
    assert execute_tool_action(action, registry=registry, context=context)["status"] == "execution_reconciliation_required"
    assert len(calls) == 1


def test_successful_tool_return_without_ledger_does_not_replay(tmp_path):
    calls = []
    def tool(**kwargs):
        calls.append(1)
        return {"status": "prepared"}
    action = {"tool": "generate_branches", "task_key": "key"}
    context = {"state_path": str(tmp_path / "state.json"), "config_version": "c", "invocation_id": "a"}
    registry = {"generate_branches": {"handler": tool}}
    assert execute_tool_action(action, registry=registry, context=context)["status"] == "completed"
    assert execute_tool_action(action, registry=registry, context=context)["status"] == "execution_reconciliation_required"
    assert len(calls) == 1


def test_missing_return_record_is_explicit_error(tmp_path):
    with pytest.raises(ValueError, match="receipt missing"):
        record_execution_return(tmp_path / "state.json", IDENTITY, "completed")

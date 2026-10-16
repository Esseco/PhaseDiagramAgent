from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock
import sqlite3
import threading

import pytest

from phase_agent.graphs.dialogue.nodes import route_request
from phase_agent.graphs.dialogue.state import TurnRuntime
from phase_agent.persistence.compact_checkpoints import CompactSqliteSaver
from tests.test_approval_history_performance import rejected_template_state


@pytest.mark.parametrize("command", ["同意", "approve", "拒绝", "reject"])
def test_empty_approval_never_routes_to_model(command, tmp_path):
    handler = SimpleNamespace(
        config_delegate=None, conversation_id=None, state_path=tmp_path / "state.json"
    )
    context = TurnRuntime(handler, [], "session")
    result = route_request({"message": command, "facts": {}}, SimpleNamespace(context=context))
    assert "reply" in result and "未调用模型" in result["reply"]
    assert context.scientific_call is None


def test_empty_approval_reuses_frozen_selection_and_restores_runtime_options(tmp_path):
    handler = SimpleNamespace(
        config_delegate=None,
        conversation_id=None,
        state_path=tmp_path / "state.json",
        workflow_kwargs={},
        _run=Mock(return_value={"status": "awaiting_approval"}),
    )
    state = rejected_template_state()
    before = deepcopy(state)
    context = TurnRuntime(handler, [], "session")
    result = route_request({"message": "同意", "facts": state}, SimpleNamespace(context=context))
    assert result["operation"] == "scientific_workflow"

    def run(invocation, feedback, message):
        assert feedback is None and message == "继续"
        assert handler.workflow_kwargs["frozen_selection_only"] is True
        return {"status": "awaiting_approval"}

    handler._run.side_effect = run
    context.scientific_call()
    handler._run.assert_called_once()
    assert handler.workflow_kwargs == {} and state == before


def test_frozen_only_cannot_fall_back_to_model_when_saved_evidence_changes():
    from phase_agent.tools.workflows.initial_tool_proposal import select_initial_proposal

    forbidden = Mock(side_effect=AssertionError("no model"))
    state = rejected_template_state()
    state["active_model_version"] = "changed"
    result = select_initial_proposal(
        current=state,
        mode="interactive",
        config={"agent": {"allowed_tools": ["select_dft_candidates"]}},
        context={"user_message": "继续", "frozen_selection_only": True},
        decision_state={},
        registry={"select_dft_candidates": {"handler": forbidden}},
        agent_client=forbidden,
        invocation_id="new",
        propose=forbidden,
        revise=forbidden,
        prepare_debug=forbidden,
        is_verified_second=forbidden,
        mc_block=forbidden,
        model_failed=forbidden,
        record_id_factory=forbidden,
    )
    assert result["status"] == "saved_selection_changed"
    forbidden.assert_not_called()


def test_pending_serialization_does_not_hold_cursor_lock(tmp_path):
    conn = sqlite3.connect(tmp_path / "checkpoint.sqlite", check_same_thread=False)
    saver = CompactSqliteSaver(conn)
    entered, release = threading.Event(), threading.Event()
    delegate = saver.serde
    errors = []

    class PausingSerde:
        def dumps_typed(self, value):
            entered.set()
            assert release.wait(5)
            return delegate.dumps_typed(value)

    saver.serde = PausingSerde()
    config = {"configurable": {"thread_id": "t", "checkpoint_ns": "", "checkpoint_id": "c"}}

    def save():
        try:
            saver.put_writes(config, [("business_frame", {"test": "value"})], "task")
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=save)
    thread.start()
    assert entered.wait(3)
    acquired = saver.lock.acquire(timeout=0.5)
    if acquired:
        saver.lock.release()
    release.set()
    thread.join(5)
    assert acquired and not thread.is_alive() and not errors
    assert saver.measurements["checkpoint_pending_write"]["calls"] == 1
    conn.close()


def test_checkpoint_shared_states_are_detached_once_and_cycles_rejected():
    from phase_agent.graphs.business_checkpoint import checkpoint_frame

    shared = {"tasks": [{"status": "pending"}]}
    saved = checkpoint_frame({"loaded_state": shared, "feedback": {"state": shared}})
    assert saved["loaded_state"] is saved["feedback"]["state"]
    shared["tasks"][0]["status"] = "completed"
    assert saved["loaded_state"]["tasks"][0]["status"] == "pending"
    cyclic = {}
    cyclic["state"] = cyclic
    with pytest.raises(ValueError, match="Circular"):
        checkpoint_frame({"loaded_state": cyclic})


@pytest.mark.parametrize("block", ["none", "active", "feedback", "training", "model", "return"])
def test_saved_input_shortcut_requires_unchanged_idle_evidence(tmp_path, monkeypatch, block):
    from phase_agent.tools.workflows.saved_dft_input_frame import prepare_saved_dft_input_frame

    state = rejected_template_state()
    state["phase_diagrams"] = {"mlip": {"status": "completed", "model_version": "m"}}
    frame = {
        "loaded_state": state,
        "execution_mode": "interactive",
        "runtime_adapters": {"frozen_selection_only": True},
        "state_path": str(tmp_path / "state.json"),
        "effective_config": {},
    }
    monkeypatch.setattr(
        "phase_agent.tools.remote.transfer_bundle.has_registered_return",
        lambda s: block == "return",
    )
    if block == "active":
        state["tasks"] = [{"task_id": "busy", "status": "pending"}]
    if block == "feedback":
        state["processed_task_ids"] = ["unprocessed"]
    if block == "training":
        state["remote_finetune_jobs"] = {"job": {"status": "pending"}}
    if block == "model":
        state["phase_diagrams"]["mlip"]["model_version"] = "old"
    result = prepare_saved_dft_input_frame(frame)
    assert bool(result.get("saved_input_only")) == (block == "none")


def test_saved_action_still_uses_project_graph_and_next_turn_resets_shortcut(tmp_path):
    from phase_agent.graphs.project.runner import run_search_workflow_graph

    fast = [True]
    observed, acted = [], []

    def initialize():
        return {
            "_workflow_prepared": True,
            "snapshot": {"config_version": "v"},
            "loaded_state": {},
            "saved_input_only": fast[0],
        }

    def collect(frame):
        observed.append(True)
        return frame

    def act(frame, **kwargs):
        acted.append(True)
        return frame

    stages = dict(
        initialize=initialize,
        collect=collect,
        analyze=lambda f: f,
        wait=lambda f: f,
        assess=lambda f: f,
        act=act,
        finalize=lambda f: {"status": "done"},
    )
    options = {
        "checkpoint_path": tmp_path / "project.sqlite",
        "project_thread": "project",
        **stages,
    }
    run_search_workflow_graph(invocation_id="first", **options)
    assert not observed and len(acted) == 1
    fast[0] = False
    run_search_workflow_graph(invocation_id="next", **options)
    assert len(observed) == 1 and len(acted) == 2


def test_pending_large_write_initializes_sqlite_before_blob_insert(tmp_path):
    conn = sqlite3.connect(tmp_path / "first.sqlite", check_same_thread=False)
    saver = CompactSqliteSaver(conn)
    config = {"configurable": {"thread_id": "t", "checkpoint_ns": "", "checkpoint_id": "c"}}
    value = {"loaded_state": {"history": ["large state"] * 50000}}
    saver.put_writes(config, [("business_frame", value)], "task")
    kind, data = conn.execute("select type,value from writes").fetchone()
    assert saver.serde.loads_typed((kind, data)) == value
    conn.close()


def test_checkpoint_state_aliases_preserve_distinct_states_and_mutable_restore(tmp_path):
    from phase_agent.graphs.business_checkpoint import restore_frame

    conn = sqlite3.connect(tmp_path / "alias.sqlite", check_same_thread=False)
    saver = CompactSqliteSaver(conn)
    shared = {"history": ["large"] * 30000}
    independent = deepcopy(shared)
    value = {
        "loaded_state": shared,
        "pre_reconciled": {"state": shared},
        "feedback": {"state": independent},
    }
    encoded = saver.serde.dumps_typed(value)
    restored = saver.serde.loads_typed(encoded)
    assert restored == value
    assert restored["loaded_state"] is restored["pre_reconciled"]["state"]
    assert restored["loaded_state"] is not restored["feedback"]["state"]
    mutable = restore_frame({"loaded_state": {}}, restored)
    mutable["loaded_state"]["history"].append("changed")
    assert len(mutable["pre_reconciled"]["state"]["history"]) == 30000
    assert len(restored["loaded_state"]["history"]) == 30000
    conn.close()


@pytest.mark.parametrize("project", ["O3-1ele-NaCoO2", "O3-1ele-NaCrO2", "O3-2ele-NaFeMn"])
def test_actual_snapshot_chat_front_door_has_zero_model_calls(tmp_path, monkeypatch, project):
    import os, json
    from pathlib import Path
    from phase_agent.runtime.chat_application import RunWorkflowChatHandler
    from phase_agent.tools.policy.file_approval import proposal_hash

    snapshot_root = Path(
        os.environ.get("PHASE_PROJECT_SNAPSHOT_DIR", Path(__file__).parents[2] / "inputs")
    )
    source = snapshot_root / f"{project}.json"
    if not source.is_file():
        pytest.skip("local snapshot not supplied")
    state = json.loads(source.read_text(encoding="utf-8"))
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    before = path.read_bytes()
    forbidden = Mock(side_effect=AssertionError("no model decision allowed"))
    handler = RunWorkflowChatHandler({"state_path": str(path), "agent_client": forbidden})
    monkeypatch.setattr(
        "phase_agent.tools.workflows.dft_selection_artifacts.proposal_artifacts",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "phase_agent.tools.workflows.generation_artifacts.proposal_artifacts", lambda *a, **k: None
    )
    pending = state.get("pending_execution_policies") or {}
    if pending:
        plan_id, saved = next(iter(pending.items()))
        handler.review_pending = Mock(return_value={"result": {"status": "completed"}})
        handler([{"role": "user", "content": "查看方案"}], conversation_id="test")
        handler([{"role": "user", "content": "同意"}], conversation_id="test")
        args = handler.review_pending.call_args
        assert args.args[:2] == (plan_id, "approve")
        assert args.kwargs["expected_proposal_hash"] == proposal_hash(saved["agent_proposal"])
    else:
        handler._run = Mock(
            return_value={"status": "not_configured", "reason": "verification only", "state": state}
        )
        handler([{"role": "user", "content": "同意"}], conversation_id="test")
        handler._run.assert_called_once()
        assert handler._run.call_args.args[1:] == (None, "继续")
    forbidden.assert_not_called()
    assert path.read_bytes() == before

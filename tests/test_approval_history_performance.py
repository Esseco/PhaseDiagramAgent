from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from phase_agent.tools.workflows.compact_action_history import compact_action_history
from phase_agent.graphs.business_checkpoint import checkpoint_frame
from phase_agent.configuration.schema.state_schema_migrations import migrate_persisted_state


class UncopyableHistory:
    def __deepcopy__(self, memo):
        raise AssertionError("discarded state must never be copied")


def test_audit_removes_states_before_copy_and_detaches_kept_fields():
    history = UncopyableHistory()
    record = {
        "state": history,
        "result": {"state": history, "tasks": ["t"]},
        "execution": {"result": {"state": history, "status": "done"}},
        "execution_result": {"result": {"state": history, "status": "done"}},
    }
    result = compact_action_history(record)
    assert result == {
        "result": {"tasks": ["t"]},
        "execution": {"result": {"status": "done"}},
        "execution_result": {"result": {"status": "done"}},
    }
    result["result"]["tasks"].append("other")
    assert record["result"]["tasks"] == ["t"]
    assert record["state"] is history


def test_checkpoint_reuses_only_content_equal_detached_fields(monkeypatch):
    frame = {
        "loaded_state": {"tasks": [{"status": "pending"}]},
        "snapshot": {"config_version": "v1"},
    }
    previous = checkpoint_frame(frame)
    import phase_agent.graphs.business_checkpoint as module

    encode = Mock(wraps=module.json.dumps)
    monkeypatch.setattr(module.json, "dumps", encode)
    same = checkpoint_frame(frame, previous)
    assert same == previous and not encode.called
    frame["loaded_state"]["tasks"][0]["status"] = "completed"
    changed = checkpoint_frame(frame, previous)
    assert changed["loaded_state"] is not previous["loaded_state"]
    assert previous["loaded_state"]["tasks"][0]["status"] == "pending"
    assert changed["loaded_state"]["tasks"][0]["status"] == "completed"
    assert changed["snapshot"] is previous["snapshot"]


def test_public_migration_and_snapshot_own_their_results():
    from phase_agent.tools.state.state_manager import update_state_snapshot

    state = {
        "tasks": [{"task_id": "t", "status": "completed"}],
        "state_snapshots": [{"history_format": "summary-v1", "nested": ["old"]}],
    }
    original = deepcopy(state)
    migrated = migrate_persisted_state(state)
    migrated["tasks"][0]["status"] = "failed"
    migrated["state_snapshots"][0]["nested"].append("new")
    assert state == original
    updated = update_state_snapshot(state)
    updated["tasks"][0]["status"] = "failed"
    updated["state_snapshots"][0]["nested"].append("new")
    assert state == original


@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_frozen_dft_selection_reuses_existing_template_gate(monkeypatch, decision):
    from phase_agent.tools.workflows.react_proposal import prepare_react_proposal

    action = {
        "tool": "select_dft_candidates",
        "parameters": {"decisions": [{"candidate_id": "s", "action": "DFT_SINGLE_POINT"}]},
    }
    proposal = {"recommended_action": "select_dft_candidates", "raw_action": action}
    stored = {"record_id": "p", "agent_proposal": deepcopy(proposal)}
    state = {"pending_execution_policies": {"p": stored}}
    client = Mock(side_effect=AssertionError("frozen proposal must not call model"))
    frame = {
        "current": state,
        "context": {},
        "invocation_id": "p",
        "config": {},
        "session": {},
        "registry": {},
        "mode": "interactive",
        "agent_client": client,
        "human_feedback": {"decision": decision},
    }
    monkeypatch.setattr(
        "phase_agent.tools.workflows.react_proposal.agent_state_summary", lambda s: {}
    )
    monkeypatch.setattr(
        "phase_agent.tools.workflows.react_proposal.stale_proposal_error", lambda *a: None
    )
    monkeypatch.setattr(
        "phase_agent.tools.workflows.react_proposal.validate_tool_action",
        lambda *a: {"valid": True},
    )
    gate = Mock(side_effect=lambda p, *args: ({**p, "template_preview": "existing gate"}, True))
    monkeypatch.setattr("phase_agent.tools.workflows.dft_template_review.gate_template", gate)
    result = prepare_react_proposal(frame)
    assert result["proposal"]["raw_action"] == proposal["raw_action"]
    client.assert_not_called()
    if decision == "approve":
        gate.assert_called_once()
        assert result["human_feedback"] is None
        assert state["pending_execution_policies"]["p"]["agent_proposal"]["template_preview"]
    else:
        gate.assert_not_called()


def test_checkpoint_paths_and_invalid_runtime_objects_unchanged(tmp_path):
    assert checkpoint_frame({"runtime_state_path": tmp_path}) == {
        "runtime_state_path": str(tmp_path)
    }
    with pytest.raises(TypeError, match="Nonserializable business field"):
        checkpoint_frame({"loaded_state": {"invalid": object()}})


def test_result_merge_copies_only_winning_history_and_preserves_consumed_approval():
    from phase_agent.tools.workflows.tool_outcomes import _apply_execution_result

    current = {"tasks": UncopyableHistory(), "keep": ["original"], "pending_execution_policies": {}}
    returned = {"tasks": [{"task_id": "t"}], "pending_execution_policies": {"p": "old"}}
    execution = {"status": "completed", "result": {"status": "prepared", "state": returned}}
    merged, status = _apply_execution_result(
        current, {"tool": "prepare_local_batch_files"}, execution, record_id="p", formal=False
    )
    assert status == "prepared"
    assert merged["pending_execution_policies"] == {}
    assert merged["tasks"] == returned["tasks"]
    merged["tasks"][0]["task_id"] = "changed"
    merged["keep"].append("new")
    assert returned["tasks"][0]["task_id"] == "t"
    assert current["keep"] == ["original"]


def rejected_template_state():
    action = {
        "tool": "select_dft_candidates",
        "task_key": "frozen-dft",
        "parameters": {"decisions": [{"candidate_id": "s", "action": "DFT_SINGLE_POINT"}]},
    }
    return {
        "confirmed_config_version": "v",
        "active_model_version": "m",
        "tasks": [],
        "action_records": [
            {
                "record_id": "old",
                "status": "rejected",
                "human_feedback": {"decision": "approve"},
                "agent_proposal": {
                    "recommended_action": "select_dft_candidates",
                    "raw_action": action,
                },
                "execution_result": {
                    "status": "completed",
                    "result": {"status": "rejected", "reason": "DFT 公共模板尚未确认", "tasks": []},
                    "operation_outcome": {
                        "config_version": "v",
                        "model_version": "m",
                        "task_ids": [],
                    },
                },
            }
        ],
    }


@pytest.mark.parametrize("change", ["none", "model", "config", "tasks", "reason", "effective"])
def test_selection_reuse_is_narrow_and_fail_closed(change):
    from phase_agent.tools.workflows.reuse_dft_selection import reusable_selection

    state = rejected_template_state()
    if change in {"model", "config"}:
        state["active_model_version" if change == "model" else "confirmed_config_version"] = (
            "changed"
        )
    elif change == "tasks":
        state["tasks"] = [{"parent_decision_id": "old"}]
    elif change == "reason":
        state["action_records"][0]["execution_result"]["result"]["reason"] = "another failure"
    elif change == "effective":
        state["effective_decisions"] = {"frozen-dft": {"status": "completed"}}
    before = deepcopy(state)
    saved = reusable_selection(state)
    assert state == before
    if change == "none":
        assert saved["raw_action"] == state["action_records"][0]["agent_proposal"]["raw_action"]
        assert saved["reuse_source_record_id"] == "old"
    else:
        assert saved is None


def test_continue_reuses_selection_without_model_or_execution():
    from phase_agent.tools.workflows.initial_tool_proposal import select_initial_proposal

    state = rejected_template_state()
    forbidden = Mock(side_effect=AssertionError("must reuse frozen selection"))
    selected = select_initial_proposal(
        current=state,
        mode="interactive",
        config={"agent": {"allowed_tools": ["select_dft_candidates"]}},
        context={"user_message": "继续"},
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
        record_id_factory=lambda *a: "new",
    )
    assert selected["record_id"] == "new"
    assert selected["proposal"]["reuse_source_record_id"] == "old"
    forbidden.assert_not_called()


def test_frozen_selection_passes_real_template_confirmation_without_reselection(
    tmp_path, monkeypatch
):
    import json
    from phase_agent.tools.workflows.react_proposal import prepare_react_proposal
    from phase_agent.tools.workflows.dft_template_review import digest, selection_hash

    state = rejected_template_state()
    state["pending_execution_policies"] = {
        "new": {
            "record_id": "new",
            "agent_proposal": deepcopy(state["action_records"][0]["agent_proposal"]),
        }
    }
    original = deepcopy(state["pending_execution_policies"]["new"]["agent_proposal"]["raw_action"])
    forbidden = Mock(side_effect=AssertionError("no reselection"))
    frame = {
        "current": state,
        "context": {},
        "invocation_id": "new",
        "config": {},
        "session": {},
        "registry": {},
        "mode": "interactive",
        "agent_client": forbidden,
        "human_feedback": {"decision": "approve"},
    }
    monkeypatch.setattr(
        "phase_agent.tools.workflows.react_proposal.agent_state_summary", lambda s: {}
    )
    monkeypatch.setattr(
        "phase_agent.tools.workflows.react_proposal.stale_proposal_error", lambda *a: None
    )
    monkeypatch.setattr(
        "phase_agent.tools.workflows.react_proposal.validate_tool_action",
        lambda *a: {"valid": True},
    )

    def preview(action, current, *args, **kwargs):
        (tmp_path / "settings.json").write_text(json.dumps({"parameters": {}}), encoding="utf-8")
        review = {
            "status": "pending",
            "config_version": "v",
            "directory": str(tmp_path),
            "selection_hash": selection_hash(action),
            "review_files": ["settings.json"],
        }
        review["digest"] = digest(review)
        return review

    render = Mock(side_effect=preview)
    monkeypatch.setattr("phase_agent.tools.workflows.dft_template_review.create_review", render)
    shown = prepare_react_proposal(frame)
    assert shown["human_feedback"] is None  # A new preview is not silently approved.
    confirmed = prepare_react_proposal({**frame, "human_feedback": {"decision": "approve"}})
    action = confirmed["proposal"]["raw_action"]
    assert action["dft_template_review"]["status"] == "approved"
    assert {k: v for k, v in action.items() if k != "dft_template_review"} == original
    render.assert_called_once()
    forbidden.assert_not_called()

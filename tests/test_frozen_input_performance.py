from unittest.mock import Mock

import pytest

from phase_agent.graphs.business_checkpoint import checkpoint_frame


class HistoryMustNotBeCopied:
    def __deepcopy__(self, memo):
        raise AssertionError("unrelated history copied")


def test_bulk_checkpoint_arrays_are_detached_and_paths_remain_json(tmp_path):
    state = {"tasks": [{"path": tmp_path, "values": (1, 2)}]}
    frame = {"loaded_state": state, "feedback": {"state": state}}
    saved = checkpoint_frame(frame)
    assert saved["loaded_state"] is saved["feedback"]["state"]
    assert saved["loaded_state"]["tasks"][0] == {"path": str(tmp_path), "values": [1, 2]}
    state["tasks"][0]["values"] = (3,)
    assert saved["loaded_state"]["tasks"][0]["values"] == [1, 2]
    assert checkpoint_frame(frame, saved)["loaded_state"]["tasks"][0]["values"] == [3]


def test_bulk_checkpoint_rejects_cycles_and_unsupported_items():
    cycle = []
    cycle.append(cycle)
    with pytest.raises(ValueError):
        checkpoint_frame({"loaded_state": {"tasks": cycle}})
    with pytest.raises(TypeError):
        checkpoint_frame({"loaded_state": {"tasks": [object()]}})


def test_comparison_evaluator_only_copies_model_bindings():
    from phase_agent.tools.workflows.create_dft_comparison_evaluator import (
        create_dft_comparison_evaluator,
    )

    state = {"history": HistoryMustNotBeCopied()}
    assert callable(create_dft_comparison_evaluator({}, state=state))
    assert "model_registry" not in state


def test_dft_validation_does_not_copy_unrelated_history():
    from phase_agent.tools.workflows.validate_dft_agent_decisions import (
        validate_dft_agent_decisions,
    )

    state = {
        "history": HistoryMustNotBeCopied(),
        "budget_reservations": {"old": {"status": "reserved"}},
    }
    result = validate_dft_agent_decisions(
        {"decisions": None}, [], state, config={}, config_version="v", remaining_budget=0
    )
    assert not result["valid"]
    assert state["budget_reservations"] == {"old": {"status": "reserved"}}


def test_frozen_summary_does_not_recompute_search_directives(monkeypatch):
    from phase_agent.tools.state.state_manager import agent_state_summary

    forbidden = Mock(side_effect=AssertionError("must not plan new search"))
    monkeypatch.setattr(
        "phase_agent.tools.policy.model_generation.new_model_generation_facts", forbidden
    )
    state = {
        "active_model_version": "m",
        "current_state_snapshot": {"decision_context": {"saved": True}},
    }
    result = agent_state_summary(state, _approval_only=True)
    result["decision_context"]["saved"] = False
    assert state["current_state_snapshot"]["decision_context"]["saved"] is True
    forbidden.assert_not_called()


def test_readonly_mc_source_does_not_copy_history_or_mutate_segments():
    from phase_agent.science.mc.second_round_state import first_round_source

    segment = {
        "branch_id": "b",
        "task_key": "t",
        "segment_index": 0,
        "model_version": "m",
        "status": "completed",
        "history": HistoryMustNotBeCopied(),
    }
    state = {"tiered_mc_state": {"segments": [segment]}, "tasks": []}
    source = first_round_source(state, "m")
    assert source and source["checksum"]
    assert "_effective_segment_index" not in segment


def test_finalize_saves_business_state_once(monkeypatch):
    from phase_agent.tools.workflows.lifecycle_finalization import _finalize_workflow

    saved = Mock()
    monkeypatch.setattr(
        "phase_agent.tools.workflows.lifecycle_finalization._save_runner_state", saved
    )
    monkeypatch.setattr(
        "phase_agent.analysis.feedback.export_dft_products.export_dft_products", lambda *a: None
    )
    monkeypatch.setattr(
        "phase_agent.tools.remote.summarize_manual_upload_wait.summarize_manual_upload_wait",
        lambda *a, **k: None,
    )
    state = {"tasks": []}
    frame = {
        "task_runner": None,
        "result": {"status": "awaiting_approval", "state": state},
        "state_path": "unused",
        "effective_config": {},
        "recovered_count": 0,
        "execution_mode": "interactive",
        "collection_report": None,
        "snapshot": {"config_version": "v"},
    }
    result = _finalize_workflow(frame)
    saved.assert_called_once_with(state, "unused")
    assert result["state"] is state and result["config_version"] == "v"


def test_dft_reservation_public_copy_and_owned_path_agree():
    from phase_agent.tools.budget.reserve_dft_actions import reserve_dft_actions

    decision = {
        "action": "DFT_SINGLE_POINT",
        "task_key": "new",
        "candidate_id": "c",
        "config_version": "v",
        "relative_cost": 2,
    }
    state = {"tasks": [{"task_id": "old"}]}
    copied = reserve_dft_actions(state, [decision])
    assert "budget_reservations" not in state
    copied["tasks"][0]["task_id"] = "changed"
    assert state["tasks"][0]["task_id"] == "old"
    owned = reserve_dft_actions(state, [decision], _owned=True)
    assert owned is state and owned["reserved_dft_cost"] == 2
    reserve_dft_actions(state, [decision], _owned=True)
    assert state["reserved_dft_cost"] == 2


def test_reconciled_mc_public_view_remains_independent():
    from phase_agent.science.mc.second_round_state import reconciled_mc_state

    state = {
        "tiered_mc_state": {"segments": [{"task_key": "t", "outputs": {"old": 1}}]},
        "tasks": [{"task_key": "t", "stage": "deep_search", "outputs": {"new": 2}}],
    }
    result = reconciled_mc_state(state)
    assert result["segments"][0]["outputs"] == {"new": 2}
    result["segments"][0]["outputs"]["new"] = 3
    assert state["tasks"][0]["outputs"]["new"] == 2
    assert state["tiered_mc_state"]["segments"][0]["outputs"] == {"old": 1}

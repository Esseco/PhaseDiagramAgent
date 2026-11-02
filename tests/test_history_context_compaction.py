from copy import deepcopy

from phase_agent.tools.workflows.compact_action_history import (
    compact_action_history,
    compact_state_history,
    compact_state_history_once,
)
from phase_agent.tools.workflows.tool_outcomes import _upsert_record


def _record():
    return {
        "record_id": "r1",
        "agent_proposal": {
            "decision_context": {"qbc_candidates": [{"candidate_id": "a"}] * 20},
            "raw_action": {"tool": "check_convergence", "parameters": {"x": 1}},
            "recommended_action": "check_convergence",
        },
        "execution_result": {"status": "completed", "result": {"value": 1}},
    }


def test_history_compaction_drops_repeated_proposal_context_without_mutating_source():
    record = _record()
    original = deepcopy(record)
    compact = compact_action_history(record)
    assert "decision_context" not in compact["agent_proposal"]
    assert compact["agent_proposal"]["raw_action"] == original["agent_proposal"]["raw_action"]
    assert record == original


def test_history_compaction_drops_large_preview_copies_but_keeps_action_identity():
    preview = {
        "task_count": 21,
        "selected_structures": [{"candidate_id": "S1"}] * 100,
        "large_pool": [{"id": str(i)} for i in range(1000)],
    }
    record = {
        "agent_proposal": {
            "action_parameters": {"relax_mc_preview": preview},
            "raw_action": {
                "tool": "prepare_local_batch_files",
                "target_ids": ["B1"],
                "parameters": {"mode": "relax_mc_inputs", "relax_mc_preview": preview},
            },
        }
    }

    compact = compact_action_history(record)

    proposal = compact["agent_proposal"]
    assert proposal["raw_action"]["tool"] == "prepare_local_batch_files"
    assert proposal["raw_action"]["target_ids"] == ["B1"]
    assert "relax_mc_preview" not in proposal["action_parameters"]
    assert "relax_mc_preview" not in proposal["raw_action"]["parameters"]


def test_state_history_compaction_is_idempotent():
    state = {
        "event_history": [
            {
                "agent_proposal": {
                    "raw_action": {
                        "parameters": {"relax_mc_preview": {"task_count": 3}}
                    }
                }
            }
        ]
    }

    assert compact_state_history_once(state) is True
    assert state["_history_compaction_version"] == 2
    assert "relax_mc_preview" not in state["event_history"][0]["agent_proposal"]["raw_action"]["parameters"]
    before = dict(state)
    assert compact_state_history_once(state) is False
    assert state == before


def test_action_record_upsert_keeps_raw_action_but_not_large_context():
    state = {"action_records": []}
    record = _record()
    _upsert_record(state, record)
    saved = state["action_records"][0]
    assert "decision_context" not in saved["agent_proposal"]
    assert saved["agent_proposal"]["raw_action"] == record["agent_proposal"]["raw_action"]


def test_legacy_history_compaction_covers_all_duplicate_receipt_collections():
    row = _record()
    state = {
        "action_records": [deepcopy(row)],
        "decisions": [deepcopy(row)],
        "event_history": [deepcopy(row)],
        "invocations": {"r1": deepcopy(row)},
    }
    compact_state_history(state, _owned=True)
    for saved in [*state["action_records"], *state["decisions"], *state["event_history"], *state["invocations"].values()]:
        assert "decision_context" not in saved["agent_proposal"]
        assert saved["agent_proposal"]["raw_action"]["tool"] == "check_convergence"

from phase_agent.persistence.models.mark_model_dependent_results_stale import (
    mark_model_dependent_results_stale,
)
from phase_agent.tools.workflows.create_model_update_handler import create_model_update_handler


def _validation():
    return {
        "status": "completed",
        "passed": True,
        "evidence_status": "research_candidate",
        "independent_test_completed": False,
        "evaluation_type": "grouped_cross_validation_review",
        "validation_data_version": "validation-v1",
    }


def test_stale_marking_copies_only_modified_collections():
    tasks = [{"task_id": "t", "payload": ["large"] * 1000}]
    diagrams = [{"id": "h", "model_version": "old", "validity": "valid"}]
    state = {
        "active_model_version": "old",
        "tasks": tasks,
        "phase_diagrams": diagrams,
        "model_transitions": [],
    }
    result = mark_model_dependent_results_stale(
        state, old_model_version="old", new_model_version="new", copy_state=False
    )
    updated = result["state"]
    assert updated is not state
    assert updated["tasks"] is tasks
    assert updated["phase_diagrams"] is not diagrams
    assert diagrams[0]["validity"] == "valid"
    assert updated["phase_diagrams"][0]["validity"] == "stale"
    assert state["active_model_version"] == "old"
    assert len(state["model_transitions"]) == 0


def test_approved_activation_avoids_recursive_copy_of_task_history():
    tasks = [{"task_id": str(i), "payload": ["large"] * 100} for i in range(20)]
    state = {
        "active_model_version": "old",
        "active_model": {"version": "old"},
        "tasks": tasks,
        "phase_diagrams": [],
        "phase_diagrams_by_model": {},
        "model_registry": {},
        "model_transitions": [],
        "model_activation_history": [],
        "candidate_models": {
            "new": {
                "status": "validated_candidate",
                "old_model_version": "old",
                "model": {"version": "new", "dataset_version": "d1"},
                "validation": _validation(),
            }
        },
    }
    result = create_model_update_handler()(
        trigger={
            "action": "ACTIVATE_CANDIDATE_MODEL",
            "candidate_model_version": "new",
            "user_approval_reason": "approved test",
        },
        state=state,
        manager=None,
        config={},
    )
    assert result["status"] == "activated"
    assert result["state"]["tasks"] is tasks
    assert result["state"]["active_model_version"] == "new"

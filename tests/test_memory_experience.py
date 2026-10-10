import pytest
from phase_agent.persistence.memory.collect_memory_candidates import collect_memory_candidates
from phase_agent.persistence.memory.review_queue import propose_knowledge_record, review_memory_update
from phase_agent.persistence.memory.retrieve_relevant_knowledge import retrieve_relevant_knowledge


def knowledge(key, **extra):
    return dict(knowledge_id=key, scope="user", category="search_rule", statement=key,
                status="active", maturity="heuristic", applicability={},
                evidence_refs=["user:instruction"], **extra)


def test_explicit_decision_outcome_preserves_original_model():
    state = {"active_model_version": "new", "tasks": [{"task_id": "t", "model_version": "old", "status": "completed"}],
             "action_records": [{"record_id": "a", "final_action": {"tool": "mc"},
                                 "execution_result": {"result": {"tasks": [{"task_id": "t"}]}}}]}
    updated = collect_memory_candidates(state)
    row = updated["memory_candidates"][0]
    assert row["model_version"] == "old"
    assert row["facts"]["outcome_status"] == "complete"
    assert len(collect_memory_candidates(updated)["memory_candidates"]) == 1
    state["action_records"][0]["execution_result"]["result"] = {}
    assert collect_memory_candidates(state)["memory_candidates"] == []


def test_superseding_requires_review_and_marks_history():
    state = {"decision_memory": {"records": [knowledge("old", approved_by_user=True)]}}
    proposed = propose_knowledge_record(state, knowledge("new", supersedes=["old"]))
    assert proposed["proposal"]["status"] == "conflict_review"
    assert retrieve_relevant_knowledge(proposed["state"])[0]["knowledge_id"] == "old"
    reviewed = review_memory_update(proposed["state"], proposed["proposal"]["proposal_id"], approved=True)["state"]
    assert reviewed["decision_memory"]["records"][0]["status"] == "superseded"
    assert retrieve_relevant_knowledge(reviewed)[0]["knowledge_id"] == "new"


def test_cross_applicability_cannot_supersede():
    state = {"decision_memory": {"records": [knowledge("old", approved_by_user=True)]}}
    proposed = propose_knowledge_record(state, {**knowledge("new", supersedes=["old"]), "applicability": {"phase": "O3"}})
    with pytest.raises(ValueError, match="same scope"):
        review_memory_update(proposed["state"], proposed["proposal"]["proposal_id"], approved=True)

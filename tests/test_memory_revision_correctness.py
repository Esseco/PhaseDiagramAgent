from copy import deepcopy
from phase_agent.persistence.memory.review_queue import (
    propose_knowledge_record,
    review_memory_update,
)
from phase_agent.persistence.memory.retrieve_relevant_knowledge import retrieve_relevant_knowledge


def row(model="v1", refs=None):
    return dict(
        scope="system",
        category="search_rule",
        statement="same statement",
        maturity="heuristic",
        source_project="p",
        system_id="s",
        applicability={},
        model_version=model,
        evidence_refs=refs or ["task:t1"],
    )


def approve(state, value):
    queued = propose_knowledge_record(state, value)
    return review_memory_update(queued["state"], queued["proposal"]["proposal_id"], approved=True)[
        "state"
    ]


def test_model_and_evidence_revision_is_saved_and_linked():
    state = {
        "tasks": [
            {"task_id": "t1", "status": "completed"},
            {"task_id": "t2", "status": "completed"},
        ]
    }
    first = approve(state, row())
    original = deepcopy(first)
    second = approve(first, row("v2", ["task:t2"]))
    assert first == original
    records = second["decision_memory"]["records"]
    assert len(records) == 2
    assert second["decision_memory"]["version"] == 2
    assert records[0]["status"] == "superseded"
    assert records[0]["knowledge_id"] in records[1]["supersedes"]
    assert records[0]["knowledge_id"] == original["decision_memory"]["records"][0]["knowledge_id"]
    assert records[0]["knowledge_family_id"] == records[1]["knowledge_family_id"]
    assert records[0]["revision_id"] != records[1]["revision_id"]


def test_changed_approved_evidence_is_not_retrieved_as_active():
    state = approve({"tasks": [{"task_id": "t1", "status": "completed"}]}, row())
    state["tasks"][0]["status"] = "quarantined"
    assert retrieve_relevant_knowledge(state, system_id="s") == []


def test_changed_evidence_same_reference_is_a_new_revision():
    state = approve({"tasks": [{"task_id": "t1", "status": "completed", "energy": 1}]}, row())
    state["tasks"][0]["energy"] = 2
    revised = approve(state, row())
    assert len(revised["decision_memory"]["records"]) == 2
    assert revised["decision_memory"]["version"] == 2


def test_persisted_revalidation_and_dependency_links():
    from phase_agent.persistence.memory.verify_evidence_refs import revalidate_memory

    state = approve({"tasks": [{"task_id": "t1", "status": "completed"}]}, row())
    identity = state["decision_memory"]["records"][0]["knowledge_id"]
    assert state["decision_memory"]["evidence_dependencies"]["task:t1"] == [identity]
    state["tasks"][0]["status"] = "invalid"
    assert revalidate_memory(state) == [identity]
    assert state["decision_memory"]["records"][0]["status"] == "needs_revalidation"
    assert revalidate_memory(state) == []


def test_candidates_revision_keeps_latest_and_original_source():
    from phase_agent.persistence.memory.collect_memory_candidates import collect_memory_candidates

    state = {
        "tasks": [{"task_id": "t", "status": "completed", "model_version": "v1"}],
        "action_records": [
            {
                "record_id": "a",
                "final_action": {"tool": "mc", "parameters": {"nested": "x" * 10000}},
                "execution_result": {"result": {"tasks": [{"task_id": "t"}]}},
            }
        ],
    }
    first = collect_memory_candidates(state)
    first["tasks"][0]["actual_cost"] = 42
    second = collect_memory_candidates(first)
    relevant = [r for r in second["memory_candidates"] if r["kind"] == "decision_outcome"]
    assert len(relevant) == 2
    assert relevant[0]["status"] == "superseded"
    assert relevant[1]["supersedes"] == [relevant[0]["candidate_id"]]
    assert "nested" not in relevant[1]["facts"]["decision"]
    assert relevant[1]["facts"]["action_parameters_ref"] == "decision_outcome:a"
    assert collect_memory_candidates(second)["memory_candidates"] == second["memory_candidates"]


def test_constraints_survive_retrieval_budget_and_query_has_scientific_scope():
    from phase_agent.persistence.memory.decision_context import (
        decision_memory_context,
        memory_query,
    )

    state = {
        "human_constraints": ["x" * 4000],
        "scientific_question": "O3 stability",
        "role_task": "force reliability",
        "failure_type": "nonconvergence",
    }
    assert decision_memory_context(state)["human_constraints"] == state["human_constraints"]
    assert all(
        word in memory_query(state)
        for word in ["O3 stability", "force reliability", "nonconvergence"]
    )


def test_approved_user_constraint_survives_tiny_advice_budget():
    value = dict(
        scope="user",
        category="human_system_knowledge",
        statement="never submit without approval " * 200,
        maturity="heuristic",
        applicability={},
        evidence_refs=["user:explicit"],
    )
    state = approve({}, value)
    constraints = []
    advice = retrieve_relevant_knowledge(state, token_budget=1, required_constraints=constraints)
    assert advice == []
    assert constraints[0]["statement"] == value["statement"].strip()


def test_team_question_and_role_tasks_reach_memory_retrieval(tmp_path, monkeypatch):
    from tests.test_codex_native_team import make_control, team
    import phase_agent.evidence.build_decision_context as builder

    queries = []
    original = builder.retrieve_relevant_knowledge

    def capture(*args, **kwargs):
        queries.append(kwargs.get("query", ""))
        return original(*args, **kwargs)

    monkeypatch.setattr(builder, "retrieve_relevant_knowledge", capture)
    control, _ = make_control(tmp_path)
    from phase_agent.tools.step_runner.file_protocol import write_json

    write_json(control.state_path, {"tasks": []})
    team(control)._context(
        question="test O3 stability", role_tasks={"reliability": {"question": "test force failure"}}
    )
    assert any("test O3 stability" in q and "test force failure" in q for q in queries)


def test_changed_pending_model_binding_cannot_keep_original_proposal_identity():
    import pytest

    proposed = propose_knowledge_record(
        {"tasks": [{"task_id": "t1", "status": "completed"}]}, row()
    )
    proposed["state"]["memory_review_queue"][0]["record"]["model_version"] = "tampered"
    with pytest.raises(ValueError, match="proposal_revision_changed"):
        review_memory_update(proposed["state"], proposed["proposal"]["proposal_id"], approved=True)
    assert proposed["state"]["memory_review_queue"][0]["status"] == "pending_review"


def test_explicitly_stale_evidence_cannot_be_newly_approved():
    import pytest

    with pytest.raises(ValueError, match="references are missing"):
        propose_knowledge_record(
            {"tasks": [{"task_id": "t1", "status": "completed", "validity": "stale"}]}, row()
        )

from phase_agent.persistence.memory.retrieve_relevant_knowledge import retrieve_relevant_knowledge
from phase_agent.tools.state.execution_receipts import begin_execution, record_execution_return, recovery_report


def record(key, text, **extra):
    return dict(knowledge_id=key, statement=text, scope="user", status="active",
                approved_by_user=True, maturity="heuristic", **extra)


def test_memory_query_and_budget():
    state = {"decision_memory": {"records": [record("a", "relax cost"), record("b", "DFT force error")]}}
    rows = retrieve_relevant_knowledge(state, query="force", limit=1)
    assert rows[0]["knowledge_id"] == "b"
    assert rows[0]["retrieval_reason"]["matched_terms"] == ["force"]
    assert retrieve_relevant_knowledge(state, token_budget=1) == []


def test_superseded_and_model_dependent_memory():
    state = {"decision_memory": {"records": [record("a", "old"),
        record("b", "new", supersedes=["a"]),
        record("c", "model", model_dependent=True, model_version="old")]}}
    assert [r["knowledge_id"] for r in retrieve_relevant_knowledge(state, model_version="new")] == ["b"]


def test_recovery_report_is_read_only_and_scoped(tmp_path):
    path = tmp_path / "state.json"
    assert recovery_report(path, {})["status"] == "clear"
    assert not (tmp_path / "execution_receipts.sqlite").exists()
    identity = dict(invocation_id="a", config_version="c", action_hash="h", tool="generate_branches")
    begin_execution(path, identity)
    assert recovery_report(path, {})["status"] == "reconciliation_required"
    assert recovery_report(tmp_path / "other.json", {})["status"] == "clear"
    completed = {"invocations": {"a": {"status": "completed", "execution": {"status": "completed"}}}}
    record_execution_return(path, identity, "completed")
    assert recovery_report(path, completed)["status"] == "clear"
    completed["invocations"]["a"]["status"] = "execution_reconciliation_required"
    assert recovery_report(path, completed)["unsettled"]

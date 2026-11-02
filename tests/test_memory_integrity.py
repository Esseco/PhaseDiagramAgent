from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from threading import RLock

import pytest

from phase_agent.runtime.local_agent_control import LocalAgentControl
from phase_agent.tools.step_runner.file_protocol import read_json, write_json
from phase_agent.persistence.memory.review_queue import (
    propose_knowledge_record,
    review_memory_update,
)
from phase_agent.persistence.memory.retrieve_relevant_knowledge import retrieve_relevant_knowledge
from phase_agent.persistence.memory.knowledge_record import knowledge_record
from phase_agent.persistence.memory.decision_context import decision_memory_context


def record(**patch):
    return {
        "scope": "system",
        "category": "search_rule",
        "statement": "test",
        "maturity": "heuristic",
        "source_project": "A",
        "system_id": "s",
        "applicability": {},
        "evidence_refs": [],
        **patch,
    }


def test_unknown_applicability_rejected():
    with pytest.raises(ValueError, match="applicability"):
        knowledge_record(record(applicability={"temperature": 1000}))


def test_project_memory_cannot_cross_project():
    state = {
        "project_id": "B",
        "decision_memory": {
            "records": [
                {
                    **record(scope="project"),
                    "knowledge_id": "K",
                    "status": "active",
                    "approved_by_user": True,
                }
            ]
        },
    }
    assert retrieve_relevant_knowledge(state, system_id="s") == []


def test_formal_context_includes_phase_memory():
    state = {
        "system_id": "s",
        "current_phase": "O3",
        "decision_memory": {
            "records": [
                {
                    **record(applicability={"phase": "O3"}),
                    "knowledge_id": "K",
                    "status": "active",
                    "approved_by_user": True,
                }
            ]
        },
    }
    assert len(decision_memory_context(state)["approved_knowledge"]) == 1


def test_evidence_is_revalidated_on_approval():
    queued = propose_knowledge_record(
        {"tasks": [{"task_id": "T", "status": "completed"}]}, record(evidence_refs=["task:T"])
    )
    queued["state"]["tasks"] = []
    with pytest.raises(ValueError, match="evidence"):
        review_memory_update(queued["state"], queued["proposal"]["proposal_id"], approved=True)


def test_running_batch_is_not_completed_evidence():
    with pytest.raises(ValueError, match="evidence"):
        propose_knowledge_record(
            {"slurm_batches": [{"batch_id": "B", "status": "running"}]},
            record(evidence_refs=["completed_batch:B"]),
        )


def test_concurrent_memory_transactions_keep_both(tmp_path, monkeypatch):
    import time
    import phase_agent.runtime.local_agent_control as module

    path = tmp_path / "state.json"
    write_json(path, {})
    control = LocalAgentControl(SimpleNamespace(state_path=path, lock=RLock()))
    original = module.propose_knowledge_record

    def slow(*args, **kwargs):
        time.sleep(0.05)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "propose_knowledge_record", slow)
    with ThreadPoolExecutor(2) as pool:
        results = list(
            pool.map(
                control.propose_memory, [record(statement="first"), record(statement="second")]
            )
        )
    assert len({r["proposal_id"] for r in results}) == 2
    assert len(read_json(path)["memory_review_queue"]) == 2


def test_changed_evidence_can_be_reproposed_as_new_revision():
    source = {"tasks": [{"task_id": "T", "status": "completed", "measurement": 1}]}
    queued = propose_knowledge_record(source, record(evidence_refs=["task:T"]))
    queued["state"]["tasks"][0]["measurement"] = 2
    renewed = propose_knowledge_record(queued["state"], record(evidence_refs=["task:T"]))
    assert renewed["proposal"]["proposal_id"] != queued["proposal"]["proposal_id"]
    approved = review_memory_update(
        renewed["state"], renewed["proposal"]["proposal_id"], approved=True
    )
    assert approved["status"] == "approved"


def test_project_knowledge_cannot_supersede_another_project():
    old = {
        **record(scope="project", source_project="A"),
        "knowledge_id": "old",
        "status": "active",
        "approved_by_user": True,
    }
    queued = propose_knowledge_record(
        {"decision_memory": {"records": [old]}},
        record(scope="project", source_project="B", supersedes=["old"]),
    )
    with pytest.raises(ValueError, match="supersedes"):
        review_memory_update(queued["state"], queued["proposal"]["proposal_id"], approved=True)


def test_project_scope_is_bound_separately_from_source_provenance():
    queued = propose_knowledge_record(
        {"project_id": "B"}, record(scope="project", source_project="A")
    )
    assert queued["proposal"]["record"]["project_id"] == "B"
    assert queued["proposal"]["record"]["source_project"] == "A"

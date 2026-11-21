"""Memory controls avoid copying private scientific history, not public input isolation."""

from copy import deepcopy
from types import SimpleNamespace
from threading import RLock

from phase_agent.runtime.local_agent_control import LocalAgentControl
from phase_agent.persistence.memory.review_queue import (
    propose_memory_update,
    propose_knowledge_record,
    review_memory_update,
)
from phase_agent.tools.step_runner.file_protocol import read_json, write_json
from tests.test_memory_integrity import record


def _forbid_history_copy(monkeypatch):
    import phase_agent.persistence.memory.review_queue as queue
    import phase_agent.persistence.memory.decision_memory as memory

    for module in (queue, memory):
        original = module.deepcopy

        def guarded(value, _original=original):
            assert not (isinstance(value, dict) and "scientific_history_probe" in value)
            return _original(value)

        monkeypatch.setattr(module, "deepcopy", guarded)


def test_memory_proposal_does_not_copy_private_history(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    history = {"large": [1] * 5000}
    write_json(path, {"system_id": "s", "scientific_history_probe": history})
    _forbid_history_copy(monkeypatch)
    control = LocalAgentControl(SimpleNamespace(state_path=path, lock=RLock()))
    result = control.propose_memory(record())
    saved = read_json(path)
    assert result["status"] == "pending_review"
    assert saved["scientific_history_probe"] == history
    assert len(saved["memory_review_queue"]) == 1


def test_memory_review_does_not_copy_private_history(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    history = {"large": [1] * 5000}
    proposed = propose_memory_update(
        {"scientific_history_probe": history},
        category="search_rules",
        items=["keep"],
        system_id="s",
        evidence_refs=[],
    )
    write_json(path, proposed["state"])
    _forbid_history_copy(monkeypatch)
    control = LocalAgentControl(SimpleNamespace(state_path=path, lock=RLock()))
    result = control.review_memory(proposed["proposal"]["proposal_id"], approved=True)
    saved = read_json(path)
    assert result["status"] == "approved" and saved["scientific_history_probe"] == history
    assert saved["decision_memory"]["long_term"]["search_rules"] == ["keep"]


def test_public_memory_calls_still_detach_input():
    state = {"system_id": "s", "scientific_history_probe": {"rows": [1]}}
    before = deepcopy(state)
    proposal = propose_knowledge_record(state, record())
    reviewed = review_memory_update(
        proposal["state"], proposal["proposal"]["proposal_id"], approved=False
    )
    proposal["state"]["scientific_history_probe"]["rows"].append(2)
    reviewed["state"]["scientific_history_probe"]["rows"].append(3)
    assert state == before


def test_private_review_error_does_not_persist_partial_approval(tmp_path):
    import pytest

    state = {"tasks": [{"task_id": "T", "status": "completed"}]}
    proposed = propose_knowledge_record(state, record(evidence_refs=["task:T"]))
    proposed["state"]["tasks"][0]["status"] = "failed"
    path = tmp_path / "state.json"
    write_json(path, proposed["state"])
    original = path.read_bytes()
    control = LocalAgentControl(SimpleNamespace(state_path=path, lock=RLock()))
    with pytest.raises(ValueError, match="evidence"):
        control.review_memory(proposed["proposal"]["proposal_id"], approved=True)
    assert path.read_bytes() == original
    assert read_json(path)["memory_review_queue"][0]["status"] == "pending_review"

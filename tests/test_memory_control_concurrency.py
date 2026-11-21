"""Reviewed memory must not overwrite or report stale cross-process state."""

import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

import phase_agent.runtime.local_agent_control as module
from phase_agent.persistence.memory.review_queue import propose_memory_update
from phase_agent.tools.step_runner.file_protocol import read_json, write_json
from tests.test_memory_integrity import record


def _concurrent_update(path):
    script = (
        "import sys; from phase_agent.tools.step_runner.file_protocol import read_json, write_json; "
        "p=sys.argv[1]; s=read_json(p); s['other_process_update']=True; write_json(p,s)"
    )
    subprocess.run([sys.executable, "-c", script, str(path)], check=True)


def _control(path):
    return module.LocalAgentControl(SimpleNamespace(state_path=path, lock=threading.RLock()))


def test_memory_proposal_does_not_overwrite_concurrent_update(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    write_json(path, {"system_id": "s"})
    original = module.propose_knowledge_record

    def competing(*args, **kwargs):
        result = original(*args, **kwargs)
        _concurrent_update(path)
        return result

    monkeypatch.setattr(module, "propose_knowledge_record", competing)
    with pytest.raises(ValueError, match="stale_state_file_version"):
        _control(path).propose_memory(record())
    saved = read_json(path)
    assert saved["other_process_update"] is True and not saved.get("memory_review_queue")


def test_memory_review_preserves_pending_on_concurrent_update(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    proposed = propose_memory_update(
        {}, category="search_rules", items=["keep"], system_id="s", evidence_refs=[]
    )
    proposal_id = proposed["proposal"]["proposal_id"]
    write_json(path, proposed["state"])
    original = module.review_memory_update

    def competing(*args, **kwargs):
        result = original(*args, **kwargs)
        _concurrent_update(path)
        return result

    monkeypatch.setattr(module, "review_memory_update", competing)
    with pytest.raises(ValueError, match="stale_state_file_version"):
        _control(path).review_memory(proposal_id, approved=True)
    saved = read_json(path)
    assert saved["other_process_update"] is True
    assert saved["memory_review_queue"][0]["status"] == "pending_review"
    assert not saved.get("decision_memory")


@pytest.mark.parametrize("has_proposal", [False, True])
def test_skill_import_rejects_stale_state(tmp_path, monkeypatch, has_proposal):
    path = tmp_path / "state.json"
    write_json(path, {})
    control = _control(path)
    monkeypatch.setattr(control, "config", lambda: {"confirmed_snapshot": {"config": {}}})
    monkeypatch.setattr(control, "_knowledge_root", lambda: tmp_path)
    monkeypatch.setattr(module, "build_system_signature", lambda _: {})

    def competing(state, root, signature):
        _concurrent_update(path)
        proposals = [{"proposal_id": "p"}] if has_proposal else []
        return {"state": {**state, "memory_review_queue": proposals}, "proposals": proposals}

    monkeypatch.setattr(module, "propose_domain_skill_import", competing)
    with pytest.raises(ValueError, match="stale_state_file_version"):
        control.propose_skill_import()
    assert read_json(path)["other_process_update"] is True


def test_memory_review_rejects_same_metadata_rewrite(tmp_path, monkeypatch):
    from tests.test_strong_state_versions import _other_process_same_metadata

    path = tmp_path / "state.json"
    proposed = propose_memory_update({"audit": "A"}, category="search_rules", items=["keep"],
                                     system_id="s", evidence_refs=[])
    proposal_id = proposed["proposal"]["proposal_id"]
    write_json(path, proposed["state"])
    original = module.review_memory_update

    def competing(*args, **kwargs):
        result = original(*args, **kwargs)
        _other_process_same_metadata(path)
        return result

    monkeypatch.setattr(module, "review_memory_update", competing)
    with pytest.raises(ValueError, match="stale_state_file_version"):
        _control(path).review_memory(proposal_id, approved=True)
    saved = read_json(path)
    assert saved["audit"] == "B"
    assert saved["memory_review_queue"][0]["status"] == "pending_review"

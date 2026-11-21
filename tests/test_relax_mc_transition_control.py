import json, threading
from types import SimpleNamespace
import pytest
from phase_agent.runtime import relax_mc_transition_control as c


def test_locked_registration_is_idempotent_and_does_not_complete_batch(tmp_path, monkeypatch):
    state_path = tmp_path / "state.json"
    state = {
        "active_model_version": "m",
        "confirmed_config_version": "c",
        "tasks": [{"task_id": "t", "status": "pending", "relax_mc_bundle_id": "b"}],
        "relax_mc_batches": {"b": {"bundle_id": "b", "status": "prepared"}},
    }
    state_path.write_text(json.dumps(state), encoding="utf8")
    control = SimpleNamespace(
        state_path=state_path,
        chat_handler=SimpleNamespace(lock=threading.RLock()),
        _codex_context_cache=None,
    )
    review = {
        "bundle_id": "b",
        "review_digest": "d",
        "status": "verified_relax_phase",
        "submitted": False,
        "task_count": 1,
        "original_failure_count": 1,
        "effective_failure_count": 0,
        "identity_check": {"mismatches": []},
    }
    monkeypatch.setattr(c, "verify_transition", lambda *a, **k: review)
    kw = dict(
        bundle_id="b",
        raw_path="raw",
        raw_sha256="x",
        receipts_path="jobs",
        receipts_sha256="y",
        expected_review_digest="d",
        user_instruction="Relax–MC threshold 3",
    )
    response = c.register(control, **kw)
    saved = json.loads(state_path.read_text(encoding="utf8"))
    assert (
        response["status"] == "registered"
        and response["submitted"] is False
        and response["mc_created"] is False
    )
    assert (
        saved["tasks"] == state["tasks"] and saved["relax_mc_batches"]["b"]["status"] == "prepared"
    )
    assert (
        saved["relax_mc_batches"]["b"]["threshold_transition"]["user_instruction"]
        == "Relax–MC threshold 3"
    )
    assert c.register(control, **kw)["status"] == "already_registered"
    with pytest.raises(ValueError, match="review_changed"):
        c.register(control, **{**kw, "expected_review_digest": "changed"})


def test_requires_explicit_instruction_and_prevents_formal_import(tmp_path, monkeypatch):
    p = tmp_path / "state.json"
    p.write_text(json.dumps({"relax_mc_batches": {"b": {"import_checksum": "x"}}}), encoding="utf8")
    ctl = SimpleNamespace(state_path=p, chat_handler=SimpleNamespace(lock=threading.RLock()))
    kw = dict(
        bundle_id="b",
        raw_path="raw",
        raw_sha256="x",
        receipts_path="jobs",
        receipts_sha256="y",
        expected_review_digest="d",
        user_instruction="",
    )
    with pytest.raises(ValueError, match="explicit_transition_instruction"):
        c.register(ctl, **kw)
    with pytest.raises(ValueError, match="formal_import_requires_review"):
        c.register(ctl, **{**kw, "user_instruction": "threshold 3"})

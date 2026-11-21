from copy import deepcopy
from hashlib import sha256

import pytest

from phase_agent.agents.strategic.codex_dft_audit_recovery import _restore_ids, recover
from phase_agent.tools.step_runner.file_protocol import write_json


def scenario(tmp_path):
    ids = ("original", "second", "replacement")
    rows = []
    for cid in ids:
        path = tmp_path / cid
        path.write_text(cid, encoding="utf8")
        rows.append(
            {
                "candidate_id": cid,
                "structure_path": str(path),
                "structure_sha256": sha256(path.read_bytes()).hexdigest(),
            }
        )
    before = {
        "active_model_version": "model",
        "confirmed_config_version": "config",
        "phase_diagrams": {
            "mlip": {"status": "completed", "version": "hull", "model_version": "model"}
        },
        "qbc_candidates": rows,
        "dft_audit_candidate_ids": ["removed", "original", "second"],
    }
    after = {"dft_audit_candidate_ids": ["replacement"]}
    action = {
        "tool": "select_dft_candidates",
        "target_ids": ["original"],
        "parameters": {"decisions": [{"candidate_id": "original", "action": "DFT_SINGLE_POINT"}]},
    }
    return before, after, action


def test_restore_uses_verified_frame_and_is_idempotent(tmp_path, monkeypatch):
    from phase_agent.agents.strategic import codex_dft_audit_recovery as module

    before, after, action = scenario(tmp_path)
    path = tmp_path / "state.json"
    current = {**before, **after, "tasks": [{"task_id": "old"}]}
    write_json(path, current)
    (tmp_path / "langgraph_lifecycle.sqlite").touch()
    proof = {"checkpoint_id": "c", "request_id": "r", "checkpoint_sha256": "x"}
    monkeypatch.setattr(
        module,
        "rejection_checkpoint",
        lambda *a, **kw: (
            proof,
            {"loaded_state": before, "result": {"state": after}},
        ),
    )
    session = {
        "team_id": "team",
        "revision": 4,
        "status": "validation_rejected",
        "submission": {
            "reason": "DFT 方案未通过筛选/预算校验，需修订后再批准：['independent_dft_audit_required']"
        },
        "events": [{"kind": "verified_validation_rejection", "revision": 4, "proof": proof}],
        "draft": {"action": action},
    }
    old_bytes = path.read_bytes()
    assert recover(path, session)["matched_target_ids"] == ["original"]
    assert path.read_bytes() == old_bytes
    assert recover(path, session, apply=True)["status"] == "restored"
    saved = module.read_json(path, {})
    assert saved["tasks"] == current["tasks"]
    assert saved["dft_audit_candidate_ids"] == ["original", "second"]
    assert saved["dft_audit_restoration"]["replaced_ids"] == ["replacement"]
    assert recover(path, session, apply=True)["status"] == "already_restored"


@pytest.mark.parametrize(
    "change, error",
    [
        ({"active_model_version": "next"}, "model_or_config_changed"),
        ({"dft_audit_candidate_ids": ["other"]}, "current_draw_changed"),
        ({"phase_diagrams": {"mlip": {"status": "completed", "version": "next"}}}, "hull_changed"),
    ],
)
def test_restore_rejects_diverged_state(tmp_path, change, error):
    before, after, action = scenario(tmp_path)
    with pytest.raises(ValueError, match=error):
        _restore_ids(before, after, {**before, **after, **change}, action)


def test_restore_rejects_changed_candidate_evidence_or_bytes(tmp_path):
    before, after, action = scenario(tmp_path)
    current = deepcopy({**before, **after})
    current["qbc_candidates"][0]["predicted_Ehull"] = 0.3
    with pytest.raises(ValueError, match="candidate_evidence_changed"):
        _restore_ids(before, after, current, action)
    (tmp_path / "original").write_text("changed", encoding="utf8")
    with pytest.raises(ValueError, match="candidate_frame_changed"):
        _restore_ids(before, after, {**before, **after}, action)

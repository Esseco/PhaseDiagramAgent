"""Serialize an explicit local quality review against current state and ledger."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from uuid import uuid4


def commit_quality_change(state_path, state, manager, transform, *, ledger_path, evidence):
    """No job execution; stale checkpoints fail closed after a data-changing review."""
    from filelock import FileLock
    from phase_agent.persistence.ledger.phase_data_manager import PhaseDataManager
    from phase_agent.tools.state.workflow_control import control_path
    from phase_agent.tools.step_runner.file_protocol import _write_json

    path, ledger = Path(state_path), Path(ledger_path)
    control = control_path(path)
    with FileLock(str(control) + ".lock", timeout=10):
        raw, ledger_raw = path.read_bytes(), ledger.read_bytes()
        saved = json.loads(raw)
        if saved.get("workflow_control_revision") != state.get(
            "workflow_control_revision"
        ) or saved.get("active_model_version") != state.get("active_model_version"):
            raise ValueError("项目状态已更新，请重新审查")
        fresh = PhaseDataManager.load(ledger)
        control_before = json.loads(control.read_bytes()) if control.exists() else None
        result = transform(saved, fresh)
        updated = result["state"]
        if (
            sha256(path.read_bytes()).digest() != sha256(raw).digest()
            or sha256(ledger.read_bytes()).digest() != sha256(ledger_raw).digest()
        ):
            raise ValueError("审查期间项目已变化，本次不覆盖")
        revision = uuid4().hex
        record = {
            "schema_version": 1,
            "state_path": str(path.resolve()),
            "revision": revision,
            "previous_revision": saved.get("workflow_control_revision"),
            "binding": {
                k: saved.get(k) for k in ("confirmed_config_version", "active_model_version")
            },
            "patch": (control_before or {}).get("patch", {}),
            "restore_allowed": False,
            "evidence": deepcopy(evidence),
        }
        updated["workflow_control_revision"] = revision
        try:
            _write_json(control, record)
            fresh.save(ledger)
            _write_json(path, updated)
        except Exception:
            _write_json(path, saved)
            _write_json(ledger, json.loads(ledger_raw))
            if control_before is None:
                control.unlink(missing_ok=True)
            else:
                _write_json(control, control_before)
            raise
        manager.data = fresh.data
    return result


def commit_human_quality_review(
    state_path, state, manager, *, review_ids, decision, user_message, ledger_path, directory=None
):
    from phase_agent.analysis.phase.phase_quality_actions import apply_quality_review
    from phase_agent.analysis.phase.phase_quality_reports import get_phase_quality_report

    if not str(user_message).strip() or decision not in {"retain", "remove", "restore"}:
        raise ValueError("需要明确人工审查决定")
    root = Path(state_path).resolve().parent.parent / "history_backups" / "phase_quality"

    def transform(saved, fresh):
        if decision == "restore":
            return apply_quality_review(
                saved,
                fresh,
                None,
                review_ids=review_ids,
                decision=decision,
                reason=user_message,
                backup_root=root,
                directory=directory,
                human_requested=True,
            )
        # Compare the numbered evidence presented this turn with the current saved view.
        presented, current = get_phase_quality_report(state), get_phase_quality_report(saved)
        if presented["input_fingerprint"] != current["input_fingerprint"]:
            raise ValueError("相图或结构已变化，请使用最新编号")
        return apply_quality_review(
            saved,
            fresh,
            current,
            review_ids=review_ids,
            decision=decision,
            reason=user_message,
            backup_root=root,
            directory=directory,
            human_requested=True,
        )

    return commit_quality_change(
        state_path,
        state,
        manager,
        transform,
        ledger_path=ledger_path,
        evidence={
            "operation": "phase_quality_human_review",
            "decision": decision,
            "review_ids": review_ids,
            "user_message": user_message,
        },
    )

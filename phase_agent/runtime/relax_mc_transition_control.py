"""Record a verified, explicitly authorized Relax-phase threshold transition."""

from pathlib import Path
from filelock import FileLock

from phase_agent.persistence.result_recovery_commit import commit_result_recovery
from phase_agent.tools.remote.transfer_bundle import checksum
from phase_agent.tools.relax_mc_batch.transition_archive import (
    verify_transition,
    register_transition,
)
from phase_agent.tools.step_runner.file_protocol import read_json


def register(
    control,
    *,
    bundle_id,
    raw_path,
    raw_sha256,
    receipts_path,
    receipts_sha256,
    expected_review_digest,
    user_instruction,
):
    """Verify under the Agent lifecycle lock; register no tasks and submit nothing."""
    if not isinstance(user_instruction, str) or not user_instruction.strip():
        raise ValueError("explicit_transition_instruction_required")
    path = Path(control.state_path).resolve()
    with (
        control.chat_handler.lock,
        FileLock(str(path.parent / "langgraph_lifecycle.sqlite") + ".lock", timeout=10),
    ):
        state = read_json(path, {}) or {}
        before_sha256 = checksum(path)
        record = (state.get("relax_mc_batches") or {}).get(bundle_id) or {}
        if record.get("import_checksum") or record.get("formal_recovery_status") == "accepted":
            raise ValueError("transition_existing_formal_import_requires_review")
        tasks = [t for t in state.get("tasks") or [] if t.get("relax_mc_bundle_id") == bundle_id]
        if not tasks or any(t.get("status") not in {"pending", "running"} for t in tasks):
            raise ValueError("transition_existing_task_state_requires_review")
        review = verify_transition(
            state, bundle_id, raw_path, raw_sha256, receipts_path, receipts_sha256
        )
        if review["review_digest"] != expected_review_digest:
            raise ValueError("transition_review_changed")
        if review["identity_check"]["mismatches"]:
            raise ValueError("transition_input_identity_requires_attestation")
        existing = record.get("threshold_transition") or {}
        if existing:
            if existing.get("review_digest") != review["review_digest"]:
                raise ValueError("transition_existing_review_conflict")
            return {
                "status": "already_registered",
                "review_digest": review["review_digest"],
                "task_count": review["task_count"],
                "submitted": False,
            }
        updated = register_transition(state, review)
        updated["relax_mc_batches"][bundle_id]["threshold_transition"]["user_instruction"] = (
            user_instruction.strip()
        )
        commit = commit_result_recovery(
            path,
            updated,
            {},
            expected_hashes={str(path): before_sha256},
            evidence={
                "operation": "register_relax_mc_threshold_transition",
                "bundle_id": bundle_id,
                "review_digest": review["review_digest"],
                "old_threshold": 1.0,
                "new_threshold": 3.0,
                "user_instruction": user_instruction.strip(),
                "submitted": False,
            },
        )
        control._codex_context_cache = None
        return {
            "status": "registered",
            "review_digest": review["review_digest"],
            "task_count": review["task_count"],
            "original_failure_count": review["original_failure_count"],
            "effective_failure_count": review["effective_failure_count"],
            "recovery_revision": commit["revision"],
            "submitted": False,
            "mc_created": False,
        }

"""Register externally observed input-byte evidence for a threshold review."""

from pathlib import Path
from filelock import FileLock
from phase_agent.persistence.result_recovery_commit import commit_result_recovery
from phase_agent.tools.remote.transfer_bundle import checksum
from phase_agent.tools.relax_mc_batch.transition_archive import (
    verify_transition,
    validate_input_attestation,
)
from phase_agent.tools.step_runner.file_protocol import read_json


def register(control, *, bundle_id, attestation_path, attestation_sha256):
    path = Path(control.state_path).resolve()
    with (
        control.chat_handler.lock,
        FileLock(str(path.parent / "langgraph_lifecycle.sqlite") + ".lock", timeout=10),
    ):
        state = read_json(path, {}) or {}
        before = checksum(path)
        record = (state.get("relax_mc_batches") or {}).get(bundle_id) or {}
        transition = record.get("threshold_transition") or {}
        if transition.get("status") != "verified_awaiting_continuation":
            raise ValueError("transition_registration_required")
        review = verify_transition(
            state,
            bundle_id,
            transition["raw_archive_path"],
            transition["raw_archive_sha256"],
            transition["scheduler_receipts_path"],
            transition["scheduler_receipts_sha256"],
        )
        if review["review_digest"] != transition.get("review_digest"):
            raise ValueError("transition_review_changed")
        receipt = validate_input_attestation(review, attestation_path, attestation_sha256)
        existing = transition.get("input_identity_attestation")
        if existing:
            if existing != receipt:
                raise ValueError("transition_input_attestation_conflict")
            return {"status": "already_registered", "count": receipt["count"], "submitted": False}
        transition["input_identity_attestation"] = receipt
        commit = commit_result_recovery(
            path,
            state,
            {},
            expected_hashes={str(path): before},
            evidence={
                "operation": "register_transition_input_attestation",
                "bundle_id": bundle_id,
                "attestation_sha256": attestation_sha256,
                "review_digest": review["review_digest"],
                "input_count": receipt["count"],
                "source": "externally_observed",
                "submitted": False,
            },
        )
        control._codex_context_cache = None
        return {
            "status": "registered",
            "count": receipt["count"],
            "review_digest": review["review_digest"],
            "revision": commit["revision"],
            "submitted": False,
        }

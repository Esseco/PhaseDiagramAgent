"""Atomically import an explicitly reviewed legacy Relax phase, without MC submission."""

from copy import deepcopy
import json
from pathlib import Path
from filelock import FileLock

from phase_agent.persistence.result_recovery_commit import commit_result_recovery
from phase_agent.persistence.ledger.phase_data_manager import PhaseDataManager
from phase_agent.tools.relax_mc_batch.transition_archive import (
    verify_transition,
    validate_input_attestation,
)
from phase_agent.tools.relax_mc_batch.transition_materialize import materialize_verified_relax
from phase_agent.tools.remote.transfer_bundle import checksum
from phase_agent.tools.state.reconcile_task_results import reconcile_task_results
from phase_agent.tools.workflows.apply_scientific_feedback import apply_scientific_feedback


def import_relax_phase(control, bundle_id):
    path = Path(control.state_path).resolve()
    dependencies = control.chat_handler.workflow_kwargs
    ledger_name = (dependencies.get("run_config") or {}).get("ledger_path") or dependencies.get(
        "ledger_path"
    )
    if not ledger_name:
        raise ValueError("transition_ledger_missing")
    ledger = Path(ledger_name).resolve()
    if not ledger.is_relative_to(path.parent / "ledgers"):
        raise ValueError("transition_ledger_outside_project")
    lock = FileLock(str(path.parent / "langgraph_lifecycle.sqlite") + ".lock", timeout=10)
    with control.chat_handler.lock, lock:
        state_sha = checksum(path)
        ledger_sha = checksum(ledger)
        state = json.loads(path.read_text(encoding="utf8"))
        record = (state.get("relax_mc_batches") or {}).get(bundle_id) or {}
        transition = record.get("threshold_transition") or {}
        if transition.get("status") == "relax_phase_imported":
            return {
                "status": "already_imported",
                "task_count": transition["task_count"],
                "submitted": False,
            }
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
        if review["review_digest"] != transition["review_digest"]:
            raise ValueError("transition_review_changed")
        if review["identity_check"]["mismatches"]:
            att = transition.get("input_identity_attestation") or {}
            if (
                validate_input_attestation(
                    review, att.get("attestation_path", ""), att.get("attestation_sha256", "")
                )
                != att
            ):
                raise ValueError("transition_attestation_changed")
        if review["effective_failure_count"]:
            raise ValueError("transition_effective_failures_require_review")
        original = [t for t in state.get("tasks") or [] if t.get("relax_mc_bundle_id") == bundle_id]
        if len(original) != review["task_count"] or any(
            t.get("status") not in {"pending", "running"} for t in original
        ):
            raise ValueError("transition_task_state_changed")
        ledger_data = json.loads(ledger.read_text(encoding="utf8"))
        manager = PhaseDataManager(ledger_data["boundary"], ledger_data.get("system_config"))
        manager.data = deepcopy(ledger_data)
        evidence_dir = (
            path.parent / "evidence" / ("relax-transition-" + review["review_digest"][:16])
        )
        results = materialize_verified_relax(
            transition["raw_archive_path"], transition["raw_archive_sha256"], evidence_dir, review
        )["results"]
        reconciliation = reconcile_task_results(state, results)
        if len(reconciliation["reconciled"]) != len(results) or any(
            row.get("status") not in {"settled", "already_settled"}
            for row in reconciliation["reconciled"]
        ):
            raise ValueError("transition_reconciliation_incomplete")
        config = reconciliation["state"].get("confirmed_config") or {}
        if (reconciliation["state"].get("shared_dft") or {}).get("enabled") or (
            config.get("shared_dft") or {}
        ).get("enabled"):
            raise ValueError("transition_shared_dft_requires_separate_transaction")
        references = dependencies.get("phase_references")
        if not references:
            raise ValueError("transition_phase_references_missing")
        feedback = apply_scientific_feedback(
            reconciliation["state"],
            results,
            manager=manager,
            active_model_version=review["model_version"],
            phase_references=references,
            phase_diagram_directory=None,
            phase_identification_cache_path=None,
        )
        state = feedback["state"]
        if len(feedback["recorded"]) != len(results) or feedback["rejected"]:
            raise ValueError("transition_scientific_feedback_incomplete")
        diagram = (state.get("phase_diagrams") or {}).get("mlip") or {}
        if (
            diagram.get("status") != "completed"
            or diagram.get("model_version") != review["model_version"]
        ):
            raise ValueError("transition_current_model_hull_unavailable")
        entry = state["relax_mc_batches"][bundle_id]
        entry["threshold_transition"].update(
            status="relax_phase_imported",
            evidence_directory=str(evidence_dir),
            phase_diagram_version=diagram.get("version"),
            imported_count=len(results),
        )
        # The original Relax–MC batch is not complete; MC has not yet run.
        entry["status"] = "partial_relax_imported"
        evidence = {
            "operation": "import_relax_mc_threshold_transition",
            "bundle_id": bundle_id,
            "review_digest": review["review_digest"],
            "raw_sha256": review["raw_archive_sha256"],
            "relax_task_count": len(results),
            "hull_version": diagram.get("version"),
            "submitted": False,
        }
        commit = commit_result_recovery(
            path,
            state,
            {str(ledger): manager.data},
            expected_hashes={str(path): state_sha, str(ledger): ledger_sha},
            evidence=evidence,
        )
        control._codex_context_cache = None
        return {
            "status": "relax_phase_imported",
            "task_count": len(results),
            "hull_version": diagram.get("version"),
            "hull_entries": len(diagram.get("entries") or []),
            "revision": commit["revision"],
            "submitted": False,
        }

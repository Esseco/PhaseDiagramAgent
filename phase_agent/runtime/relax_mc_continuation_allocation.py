"""Register an approved Relax–MC batch's remaining MC wave."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import zipfile
from filelock import FileLock
from phase_agent.persistence.ledger.phase_data_manager import PhaseDataManager
from phase_agent.persistence.result_recovery_commit import commit_result_recovery
from phase_agent.tools.relax_mc_batch.analysis import next_mc
from phase_agent.tools.relax_mc_batch.branch_limit import validate_branch_scope
from phase_agent.tools.relax_mc_batch.transition_archive import (
    verify_transition,
    validate_input_attestation,
)
from phase_agent.tools.remote.integrity import file_checksum
from phase_agent.tools.remote.transfer_bundle import checksum


def allocate(control, *, bundle_id, preview_path, preview_sha256):
    """Reuse exactly the existing approved batch; create no HPC jobs or input packages."""
    state_path = Path(control.state_path).resolve()
    deps = control.chat_handler.workflow_kwargs
    ledger_path = Path(
        (deps.get("run_config") or {}).get("ledger_path") or deps["ledger_path"]
    ).resolve()
    with (
        control.chat_handler.lock,
        FileLock(str(state_path.parent / "langgraph_lifecycle.sqlite") + ".lock", timeout=10),
    ):
        before = checksum(state_path)
        ledger_before = checksum(ledger_path)
        state = json.loads(state_path.read_text(encoding="utf8"))
        record = (state.get("relax_mc_batches") or {}).get(bundle_id) or {}
        transition = record.get("threshold_transition") or {}
        previous = transition.get("mc_first_wave") or {}
        if previous:
            if previous.get("preview_sha256") != preview_sha256:
                raise ValueError("continuation_prior_preview_conflict")
            return {
                "status": "already_allocated",
                "task_count": len(previous["task_ids"]),
                "submitted": False,
            }
        if (
            transition.get("status") != "relax_phase_imported"
            or record.get("status") != "partial_relax_imported"
        ):
            raise ValueError("continuation_imported_relax_required")
        if state.get("active_model_version") != record.get("model_version") or state.get(
            "confirmed_config_version"
        ) != record.get("config_version"):
            raise ValueError("continuation_current_versions_changed")
        records = [
            r
            for r in state.get("action_records") or []
            if (r.get("final_action") or {})
            .get("parameters", {})
            .get("relax_mc_preview", {})
            .get("bundle_id")
            == bundle_id
        ]
        if (
            len(records) != 1
            or (records[0].get("human_feedback") or {}).get("decision") != "approve"
        ):
            raise ValueError("continuation_original_approval_missing")
        original = records[0]["final_action"]
        frozen = original["parameters"]["relax_mc_preview"]
        if (
            original.get("tool") != "prepare_local_batch_files"
            or original["parameters"].get("mode") != "relax_mc_inputs"
            or frozen.get("checksum") != record.get("plan_checksum")
            or set(original.get("target_ids") or []) != set(record["branch_ids"])
        ):
            raise ValueError("continuation_original_approval_scope_changed")
        validate_branch_scope(record["branch_ids"])
        if any(
            t.get("relax_mc_bundle_id") == bundle_id and t.get("stage") == "deep_search"
            for t in state.get("tasks") or []
        ):
            raise ValueError("continuation_mc_already_exists")
        review = verify_transition(
            state,
            bundle_id,
            transition["raw_archive_path"],
            transition["raw_archive_sha256"],
            transition["scheduler_receipts_path"],
            transition["scheduler_receipts_sha256"],
        )
        if review["review_digest"] != transition["review_digest"]:
            raise ValueError("continuation_review_changed")
        if review["identity_check"]["mismatches"]:
            att = transition.get("input_identity_attestation") or {}
            if (
                validate_input_attestation(
                    review, att.get("attestation_path", ""), att.get("attestation_sha256", "")
                )
                != att
            ):
                raise ValueError("continuation_input_attestation_changed")
        preview_file = Path(preview_path)
        if checksum(preview_file) != preview_sha256:
            raise ValueError("continuation_preview_changed")
        expected = json.loads(preview_file.read_text(encoding="utf8"))
        diagram = (state.get("phase_diagrams") or {}).get("mlip") or {}
        if (
            expected.get("bundle_id") != bundle_id
            or expected.get("review_digest") != review["review_digest"]
            or expected.get("hull_version") != diagram.get("version")
            or diagram.get("version") != transition.get("phase_diagram_version")
            or expected.get("submitted") is not False
            or expected.get("project_tasks_created") is not False
        ):
            raise ValueError("continuation_preview_binding_invalid")
        ledger = json.loads(ledger_path.read_text(encoding="utf8"))
        manager = PhaseDataManager(ledger["boundary"], ledger.get("system_config"))
        manager.data = deepcopy(ledger)
        manager.data["branches"] = {
            bid: manager.data["branches"][bid] for bid in record["branch_ids"]
        }
        with zipfile.ZipFile(record["upload_archive"]) as archive:
            manifest = json.loads(archive.read("batch.json"))
        if any(
            manifest.get(k) != record.get(k)
            for k in ("bundle_id", "plan_checksum", "model_version", "config_version")
        ):
            raise ValueError("continuation_frozen_batch_changed")
        with tempfile.TemporaryDirectory(prefix="phase-mc-allocation-") as directory:
            plan = next_mc(state, manager, manifest, Path(directory), 0)
        actions = plan.get("actions") or []
        if (
            plan.get("status") != "scheduled"
            or actions != expected.get("actions")
            or plan.get("missing_branch_ids", []) != expected.get("missing_branch_ids", [])
        ):
            raise ValueError("continuation_allocation_changed")
        if len(actions) > frozen["workload"]["mc_task_upper"][0] or len(
            {a["branch_id"] for a in actions}
        ) != len(actions):
            raise ValueError("continuation_exceeds_approved_scope")
        by_structure = {r["structure_id"]: r for r in review["rows"]}
        for action in actions:
            row = by_structure.get(action.get("structure_id"))
            if (
                not row
                or action.get("branch_id") != row["branch_id"]
                or file_checksum(action["structure_path"]) != row["final_sha256"]
            ):
                raise ValueError("continuation_frame_binding_invalid")
        next_state = plan["state"]
        tasks = [
            t
            for t in next_state["tasks"]
            if t.get("relax_mc_bundle_id") == bundle_id and t.get("stage") == "deep_search"
        ]
        if len(tasks) != len(actions) or any(t.get("status") != "pending" for t in tasks):
            raise ValueError("continuation_registered_tasks_invalid")
        next_state["relax_mc_batches"][bundle_id]["threshold_transition"]["mc_first_wave"] = {
            "status": "allocated_not_submitted",
            "preview_sha256": preview_sha256,
            "task_ids": sorted(t["task_id"] for t in tasks),
            "original_action_record_id": records[0]["record_id"],
            "hull_version": diagram["version"],
        }
        if checksum(ledger_path) != ledger_before:
            raise ValueError("continuation_ledger_changed")
        result = commit_result_recovery(
            state_path,
            next_state,
            {},
            expected_hashes={str(state_path): before},
            evidence={
                "operation": "allocate_approved_relax_mc_continuation",
                "bundle_id": bundle_id,
                "preview_sha256": preview_sha256,
                "hull_version": diagram["version"],
                "task_count": len(tasks),
                "submitted": False,
            },
        )
        control._codex_context_cache = None
        return {
            "status": "allocated",
            "task_count": len(tasks),
            "hull_version": diagram["version"],
            "revision": result["revision"],
            "submitted": False,
        }

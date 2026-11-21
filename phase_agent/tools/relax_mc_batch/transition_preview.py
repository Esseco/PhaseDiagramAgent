"""Preview the next MC wave from a verified legacy Relax threshold transition."""

from copy import deepcopy
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

from phase_agent.tools.relax_mc_batch.transition_archive import verify_transition
from phase_agent.tools.relax_mc_batch.transition_effective import effective_relax_result
from phase_agent.tools.remote.transfer_bundle import checksum, import_bundle
from phase_agent.tools.relax_mc_batch.upload_driver import prepare_work
from phase_agent.tools.relax_mc_batch.files import read, rebase
from phase_agent.tools.remote.integrity import verify_result
from phase_agent.tools.remote.resolve_local_relax_structure import resolve_local_relax_structure
from phase_agent.tools.state.normalize_task_result import normalize_task_result
from phase_agent.persistence.ledger.phase_data_manager import PhaseDataManager
from phase_agent.tools.relax_mc_batch.analysis import feedback, next_mc


def preview(state, bundle_id):
    """Compute an isolated, non-submitting preview; never write into the project."""
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
    if review["identity_check"]["mismatches"]:
        from phase_agent.tools.relax_mc_batch.transition_archive import validate_input_attestation

        receipt = transition.get("input_identity_attestation") or {}
        checked = validate_input_attestation(
            review, receipt.get("attestation_path", ""), receipt.get("attestation_sha256", "")
        )
        if checked != receipt:
            raise ValueError("transition_input_identity_attestation_changed")
    if review["effective_failure_count"]:
        raise ValueError("transition_has_new_threshold_failures")
    with tempfile.TemporaryDirectory(prefix="phase-transition-preview-") as temporary:
        root = Path(temporary)
        input_folder = root / "input"
        input_folder.mkdir()
        shutil.copyfile(record["upload_archive"], input_folder / "inputs.zip")
        work = prepare_work(input_folder)
        raw_root = root / "returned"
        import_bundle(
            transition["raw_archive_path"], raw_root, extra_result_suffixes=(".csv", ".bin")
        )
        manifest = rebase(read(work / "batch.json"), work)
        source = rebase(read(raw_root / "workflow_state.json"), raw_root)
        # Old frozen packages used Windows absolute reference paths. Resolve
        # only files included in the verified package and retain their basename.
        references = {}
        for phase, old in manifest.get("phase_references", {}).items():
            name = PurePosixPath(str(old).replace("\\", "/")).name
            candidate = (work / "references" / name).resolve()
            if not candidate.is_file() or not candidate.is_relative_to(
                (work / "references").resolve()
            ):
                raise ValueError("transition_phase_reference_missing:" + phase)
            references[phase] = str(candidate)
        manifest["phase_references"] = references
        data = manifest["manager"]
        manager = PhaseDataManager(data["boundary"], data.get("system_config"))
        manager.data = data
        working = deepcopy(manifest["initial_state"])
        rows = {r["task_id"]: r for r in review["rows"]}
        results = []
        for task in source["tasks"]:
            path = Path(task["result_path"])
            checked = verify_result(path, path.with_name("task.finished.json"), task)
            if not checked["valid"] or checksum(path) != rows[task["task_id"]]["result_sha256"]:
                raise ValueError("transition_result_changed:" + task["task_id"])
            view = effective_relax_result(checked["result"], rows[task["task_id"]])
            view = resolve_local_relax_structure(view, path.parent)
            if view is None:
                raise ValueError("transition_final_structure_unavailable:" + task["task_id"])
            results.append(
                normalize_task_result(
                    {**task, "object_id": task.get("object_id", task["structure_id"])}, view
                )
            )
        working = feedback(working, results, manager, references, work, manifest["model_version"])
        plan = next_mc(working, manager, manifest, work, 0)
        actions = plan.get("actions") or []
        source_paths = {
            str(Path(task["result_path"]).parent / "initial_relaxed.vasp"): task
            for task in source["tasks"]
        }
        portable_actions = []
        for action in actions:
            path = Path(action["structure_path"]).resolve()
            task = source_paths.get(str(path))
            if task is None or not path.is_relative_to(raw_root.resolve()):
                raise ValueError("transition_mc_source_unbound:" + str(action.get("branch_id")))
            row = rows[task["task_id"]]
            if checksum(path) != row["final_sha256"] or task["structure_id"] != action.get(
                "structure_id"
            ):
                raise ValueError("transition_mc_source_changed:" + task["task_id"])
            item = {k: v for k, v in action.items() if k != "structure_path"}
            item["structure_source"] = {
                "raw_archive_path": transition["raw_archive_path"],
                "raw_archive_sha256": transition["raw_archive_sha256"],
                "archive_member": path.relative_to(raw_root.resolve()).as_posix(),
                "sha256": row["final_sha256"],
                "source_task_id": task["task_id"],
            }
            portable_actions.append(item)
        return {
            "status": plan["status"],
            "reason": plan.get("reason"),
            "bundle_id": bundle_id,
            "review_digest": review["review_digest"],
            "relax_task_count": len(results),
            "mc_action_count": len(portable_actions),
            "actions": portable_actions,
            "missing_branch_ids": plan.get("missing_branch_ids", []),
            "excluded_candidates": plan.get("excluded_candidates", []),
            "model_version": manifest["model_version"],
            "config_version": manifest["config_version"],
            "submitted": False,
            "project_tasks_created": False,
        }

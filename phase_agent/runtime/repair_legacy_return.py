"""Explicit, local identity-checked recovery of unbound legacy failure returns."""

import argparse
import json
from pathlib import Path
import tempfile
import zipfile

from phase_agent.tools.remote.legacy_recovery import recovery_plan
from phase_agent.tools.remote.legacy_recovery_runner import artifacts, terminal
from phase_agent.tools.remote.transfer_bundle import checksum, create_bundle, import_bundle


def repair_failure_return(state, archive, receipt, *, task_id, job_id, include_completed=False):
    archive, receipt = Path(archive).resolve(), Path(receipt).resolve()
    matches = [b for b in state.get("slurm_batches", []) if task_id in b.get("task_ids", [])]
    if len(matches) != 1:
        raise ValueError("ambiguous_registered_task")
    handoff = matches[0]["handoff"]
    if archive.parent != Path(handoff["return_directory"]).resolve():
        raise ValueError("return_not_in_registered_directory")
    plan = recovery_plan(state, handoff)
    task = next(t for t in plan["tasks"] if t["expected"]["task_id"] == task_id)
    submitted = json.loads(receipt.read_text(encoding="utf-8"))
    if submitted.get("archive_sha256") != plan["source_archive_sha256"]:
        raise ValueError("submission_input_checksum_mismatch")
    script = task["input_directory"] + "/GPU.sh"
    if str((submitted.get("jobs", {}).get(script) or {}).get("job_id")) != str(job_id):
        raise ValueError("submission_job_mismatch")
    with zipfile.ZipFile(archive) as zipped:
        manifest = json.loads(zipped.read("transfer_manifest.json"))
        if manifest.get("batch_binding") is not None or manifest.get("kind") != "results":
            raise ValueError("expected_unbound_legacy_results")
    label = "terminal" if include_completed else "failure"
    destination = archive.with_name(f"verified-{label}-{job_id}.results.zip")
    report_path = archive.with_name(f"verified-{label}-{job_id}.json")
    if destination.exists() or report_path.exists():
        raise FileExistsError("repair_already_exists; inspect_saved_receipt")
    with tempfile.TemporaryDirectory(prefix="legacy-return-") as temporary:
        staging = Path(temporary)
        import_bundle(archive, staging)
        directory = staging / task["result_directory"]
        data = terminal(directory, task["expected"])
        allowed = (
            {"completed", "failed", "timeout", "cancelled"}
            if include_completed
            else {"failed", "timeout"}
        )
        if data is None or data.get("status") not in allowed:
            raise ValueError("expected_terminal_failure")
        if str((data.get("runtime_observation") or {}).get("job_id")) != str(job_id):
            raise ValueError("result_job_mismatch")
        # Only the verified failed task is re-exported; no scientific labels are altered.
        files = (
            list(artifacts(staging, directory, task, data).values())
            if include_completed
            else [directory / "result.json", directory / "task.finished.json"]
        )
        create_bundle(
            staging,
            files,
            destination,
            kind="results",
            binding={
                "batch": plan["source_folder"].removesuffix("_input"),
                "input_sha256": plan["source_archive_sha256"],
            },
        )
    report = {
        "status": "verified",
        "task_id": task_id,
        "job_id": str(job_id),
        "source_archive": str(archive),
        "source_sha256": checksum(archive),
        "submission_receipt": str(receipt),
        "submission_sha256": checksum(receipt),
        "input_sha256": plan["source_archive_sha256"],
        "repaired_archive": str(destination),
        "repaired_sha256": checksum(destination),
        "science_submissions": 0,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def repair_terminal_returns(state, archive, receipt):
    """Verify each terminal task against the original receipt, without changing State."""
    submitted = json.loads(Path(receipt).read_text(encoding="utf-8"))
    with zipfile.ZipFile(archive) as zipped:
        tasks = [
            json.loads(zipped.read(n)) for n in zipped.namelist() if n.endswith("/result.json")
        ]
    reports = []
    for task in tasks:
        task_id = task["task_id"]
        job_id = str((task.get("runtime_observation") or {}).get("job_id"))
        if not any(str(row.get("job_id")) == job_id for row in submitted.get("jobs", {}).values()):
            raise ValueError("terminal_job_not_in_submission_receipt")
        report_file = Path(archive).with_name(f"verified-terminal-{job_id}.json")
        if report_file.is_file():
            saved = json.loads(report_file.read_text(encoding="utf-8"))
            if saved["source_sha256"] != checksum(archive) or saved["repaired_sha256"] != checksum(
                saved["repaired_archive"]
            ):
                raise ValueError("saved_terminal_repair_changed")
            reports.append(saved)
            continue
        reports.append(
            repair_failure_return(
                state, archive, receipt, task_id=task_id, job_id=job_id, include_completed=True
            )
        )
    return {"status": "verified", "task_count": len(reports), "reports": reports}



def acknowledge_verified_source(state, archive, receipt):
    """Import an old unbound source only after every terminal task was verified and imported."""
    from phase_agent.tools.remote.transfer_bundle import _archive_receipt

    archive, receipt = Path(archive).resolve(), Path(receipt).resolve()
    batches = [b for b in state.get("slurm_batches", [])
               if (b.get("handoff") or {}).get("return_directory")
               and Path(b["handoff"]["return_directory"]).resolve() == archive.parent]
    if not batches:
        raise ValueError("return_not_in_registered_directory")
    handoff = batches[0]["handoff"]
    if any(b["handoff"] != handoff for b in batches):
        raise ValueError("ambiguous_registered_handoff")
    plan = recovery_plan(state, handoff)
    submitted = json.loads(receipt.read_text(encoding="utf8"))
    if submitted.get("archive_sha256") != plan["source_archive_sha256"]:
        raise ValueError("submission_input_checksum_mismatch")
    source_sha, submission_sha = checksum(archive), checksum(receipt)
    processed = set(state.get("processed_task_ids") or [])
    feedback = set(state.get("feedback_processed_task_ids") or [])
    registered = {t for batch in batches for t in batch.get("task_ids", [])}
    expected_ids = {t["expected"]["task_id"] for t in plan["tasks"]}
    if not expected_ids or registered != expected_ids or not expected_ids <= processed & feedback:
        raise ValueError("legacy_terminal_feedback_incomplete")
    with zipfile.ZipFile(archive) as zipped:
        manifest = json.loads(zipped.read("transfer_manifest.json"))
        if manifest.get("batch_binding") is not None or manifest.get("kind") != "results":
            raise ValueError("expected_unbound_legacy_results")
    with tempfile.TemporaryDirectory(prefix="legacy-source-audit-") as temporary:
        staging = Path(temporary)
        import_bundle(archive, staging)
        found = {str(p.parent.relative_to(staging)).replace("\\", "/")
                 for p in staging.rglob("result.json")}
        if found != {task["result_directory"] for task in plan["tasks"]}:
            raise ValueError("legacy_result_scope_mismatch")
        for task in plan["tasks"]:
            task_id = task["expected"]["task_id"]
            result = terminal(staging / task["result_directory"], task["expected"])
            if result is None or result.get("status") not in {"completed", "failed", "timeout", "cancelled"}:
                raise ValueError("legacy_task_not_terminal")
            job_id = str((result.get("runtime_observation") or {}).get("job_id"))
            script = task["input_directory"] + "/GPU.sh"
            if str((submitted.get("jobs", {}).get(script) or {}).get("job_id")) != job_id:
                raise ValueError("submission_job_mismatch")
            report_file = archive.with_name(f"verified-terminal-{job_id}.json")
            report = json.loads(report_file.read_text(encoding="utf8"))
            repaired = Path(report["repaired_archive"]).resolve()
            repaired_receipt = _archive_receipt(archive.parent, repaired)
            if (report.get("status") != "verified" or report.get("task_id") != task_id
                or str(report.get("job_id")) != job_id or report.get("source_sha256") != source_sha
                or report.get("submission_sha256") != submission_sha
                or report.get("input_sha256") != plan["source_archive_sha256"]
                or repaired.parent != archive.parent or not repaired.is_file()
                or checksum(repaired) != report.get("repaired_sha256")
                or not repaired_receipt.is_file()
                or repaired_receipt.read_text(encoding="utf8") != report.get("repaired_sha256")):
                raise ValueError("verified_terminal_evidence_mismatch")
        destination = Path(handoff["import_directory"])
        for path in staging.rglob("*"):
            if not path.is_file():
                continue
            target = destination / path.relative_to(staging)
            if not target.exists() and not (path.name.startswith("gpu-job-") and path.name.endswith(".runtime.json")):
                raise ValueError("legacy_scientific_file_not_imported")
        import_bundle(archive, destination)
    target_receipt = _archive_receipt(archive.parent, archive)
    target_receipt.parent.mkdir(exist_ok=True)
    target_receipt.write_text(source_sha, encoding="utf8")
    return {"status": "verified_and_receipted", "task_count": len(expected_ids),
            "archive": str(archive), "sha256": source_sha, "receipt": str(target_receipt),
            "science_submissions": 0}



def acknowledge_verified_subset(state, archive, reference):
    """Import an unbound old subset only if every member matches a receipted source."""
    from phase_agent.tools.remote.transfer_bundle import _archive_receipt, _validated_inventory, MAX_ARCHIVE_BYTES

    archive, reference = Path(archive).resolve(), Path(reference).resolve()
    if archive == reference or archive.parent != reference.parent:
        raise ValueError("legacy_subset_source_mismatch")
    batches = [b for b in state.get("slurm_batches", [])
               if (b.get("handoff") or {}).get("return_directory")
               and Path(b["handoff"]["return_directory"]).resolve() == archive.parent]
    if not batches or any(b["handoff"] != batches[0]["handoff"] for b in batches):
        raise ValueError("ambiguous_registered_handoff")
    receipt = _archive_receipt(reference.parent, reference)
    if not receipt.is_file() or receipt.read_text(encoding="utf8") != checksum(reference):
        raise ValueError("reference_source_not_receipted")
    with zipfile.ZipFile(archive) as subset, zipfile.ZipFile(reference) as original:
        if json.loads(subset.read("transfer_manifest.json")).get("batch_binding") is not None:
            raise ValueError("expected_unbound_legacy_results")
        rows = _validated_inventory(subset, MAX_ARCHIVE_BYTES, None)
        original_rows = {r["path"]: r for r in _validated_inventory(original, MAX_ARCHIVE_BYTES, None)}
        if not rows or any(r != original_rows.get(r["path"]) for r in rows):
            raise ValueError("legacy_subset_inventory_mismatch")
    import_bundle(archive, batches[0]["handoff"]["import_directory"])
    target = _archive_receipt(archive.parent, archive)
    target.parent.mkdir(exist_ok=True)
    target.write_text(checksum(archive), encoding="utf8")
    return {"status": "verified_and_receipted", "files": len(rows), "archive": str(archive),
            "reference": str(reference), "receipt": str(target), "science_submissions": 0}



def reconcile_verified_legacy_returns(state, errors):
    """During recovery, settle only old unbound archives backed by full evidence."""
    pending = [dict(row) for row in errors]
    roots = {Path(row["archive"]).parent for row in pending
             if row.get("reason") == "returned_package_binding_mismatch" and row.get("archive")}
    references = {}
    for root in roots:
        for report_file in root.glob("verified-terminal-*.json"):
            try:
                report = json.loads(report_file.read_text(encoding="utf8"))
                source = Path(report["source_archive"]).resolve()
                submitted = Path(report["submission_receipt"]).resolve()
                if source.parent == root.resolve() and submitted.parent == root.resolve():
                    references[source] = submitted
            except (OSError, ValueError, KeyError, TypeError):
                continue
    for reference, submitted in references.items():
        for row in list(pending):
            if row.get("reason") != "returned_package_binding_mismatch" or Path(row["archive"]).resolve() != reference:
                continue
            try:
                acknowledge_verified_source(state, reference, submitted)
            except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
                row["legacy_repair_blocked"] = str(error)
            else:
                pending.remove(row)
    for row in list(pending):
        if row.get("reason") != "returned_package_binding_mismatch":
            continue
        archive = Path(row["archive"]).resolve()
        for reference in references:
            if archive == reference or archive.parent != reference.parent:
                continue
            try:
                acknowledge_verified_subset(state, archive, reference)
            except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile):
                continue
            pending.remove(row)
            break
    return pending


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--receipt")
    parser.add_argument("--reference-source")
    parser.add_argument("--task-id")
    parser.add_argument("--job-id")
    parser.add_argument("--all-terminal", action="store_true")
    parser.add_argument("--acknowledge-verified-source", action="store_true")
    args = parser.parse_args(argv)
    if not args.reference_source and not args.receipt:
        parser.error("--receipt is required unless --reference-source is given")
    from phase_agent.runtime.codex.offline import CodexOffline

    codex = CodexOffline(args.runtime_config)
    with codex.handler.lock:
        state = json.loads(Path(codex.handler.state_path).read_text(encoding="utf-8"))
        if args.reference_source:
            result = acknowledge_verified_subset(state, args.archive, args.reference_source)
        elif args.acknowledge_verified_source:
            result = acknowledge_verified_source(state, args.archive, args.receipt)
        elif args.all_terminal:
            result = repair_terminal_returns(state, args.archive, args.receipt)
        else:
            if not args.task_id or not args.job_id:
                parser.error("--task-id and --job-id are required without --all-terminal")
            result = repair_failure_return(
                state, args.archive, args.receipt, task_id=args.task_id, job_id=args.job_id
            )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

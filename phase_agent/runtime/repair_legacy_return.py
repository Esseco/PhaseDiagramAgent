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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--task-id")
    parser.add_argument("--job-id")
    parser.add_argument("--all-terminal", action="store_true")
    args = parser.parse_args(argv)
    from phase_agent.runtime.codex.offline import CodexOffline

    codex = CodexOffline(args.runtime_config)
    with codex.handler.lock:
        state = json.loads(Path(codex.handler.state_path).read_text(encoding="utf-8"))
        if args.all_terminal:
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

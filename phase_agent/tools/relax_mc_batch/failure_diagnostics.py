"""Bounded remote failure evidence; no inference, submission or task settlement."""

from collections import Counter
from pathlib import Path
import json
import re


def unverified_task_ids(state, expected=None):
    tasks = {task["task_id"]: task for task in state.get("tasks") or []}
    expected = set(tasks) if expected is None else set(expected)
    errors = set(state.get("result_validation_errors") or {}) | set(
        state.get("scientific_feedback_errors") or {}
    )
    return {
        tid
        for tid in expected
        if tid in errors
        or tasks.get(tid, {}).get("status") not in {"completed", "failed", "timeout", "cancelled"}
    }


def log_tails(root, batch):
    root = Path(root).resolve()
    script = Path(batch.get("script_path") or root / "missing-script")
    job = str(batch.get("job_id") or "unknown")
    paths = {script.parent / f"slurm-{job}.out", root / f"slurm-{job}.out"}
    if script.is_file() and script.resolve().is_relative_to(root):
        content = script.read_text(encoding="utf-8", errors="replace")
        name = re.search(r"^#SBATCH --job-name=(.+)$", content, re.M)
        for value in re.findall(r"^#SBATCH --(?:output|error)=(.+)$", content, re.M):
            value = value.strip().strip("\"'").replace("%j", job).replace("%A", job)
            if name:
                value = value.replace("%x", name[1].strip())
            path = Path(value)
            paths.add(path if path.is_absolute() else script.parent / path)
    rows = []
    for path in sorted(paths):
        if not path.resolve().is_relative_to(root) or not path.is_file():
            continue
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 8192))
            rows.append(
                {
                    "path": str(path.relative_to(root)),
                    "log_tail": stream.read().decode("utf-8", "replace"),
                }
            )
        if len(rows) == 3:
            break
    return rows


def record_remote_diagnostics(root, state, results, missing, report, io):
    missing = set(missing)
    failed = [r for r in results if r.get("status") in {"failed", "timeout", "cancelled"}]
    affected = missing | {r["task_id"] for r in failed}
    batches = [
        b for b in state.get("slurm_batches", []) if affected.intersection(b.get("task_ids") or [])
    ]
    jobs = [str(b["job_id"]) for b in batches if b.get("job_id")]
    evidence = io.evidence(jobs) if jobs and callable(getattr(io, "evidence", None)) else {}
    by_task, job_rows = {}, []
    for batch in batches:
        job = str(batch.get("job_id") or "unknown")
        row = {"job_id": job, "state": "UNKNOWN", "exit_code": None, **evidence.get(job, {})}
        try:
            row["logs"] = log_tails(root, batch)
        except OSError as error:
            row["logs"] = []
            row["log_error"] = str(error)[:1024]
        job_rows.append(row)
        by_task.update({tid: job for tid in batch.get("task_ids") or []})
    issues = [
        {
            "origin": "remote_calculation",
            "code": "worker_" + r["status"],
            "task_id": r["task_id"],
            "job_id": by_task.get(r["task_id"]),
            "error": str(
                r.get("error") or r.get("failure_reason") or "worker returned " + r["status"]
            )[:2048],
            "traceback": str(r.get("traceback") or "")[-4096:],
        }
        for r in failed
    ]
    details = {r["task_id"]: r["reason"] for r in report.get("details_preview") or []}
    for tid in sorted(missing):
        job = by_task.get(tid)
        outcome = evidence.get(job, {})
        validation = (state.get("result_validation_errors") or {}).get(tid) or (
            state.get("scientific_feedback_errors") or {}
        ).get(tid)
        failed_job = outcome.get("state") in {
            "FAILED",
            "TIMEOUT",
            "OUT_OF_MEMORY",
            "NODE_FAIL",
            "CANCELLED",
        }
        issues.append(
            {
                "origin": "remote_validation"
                if validation
                else "remote_job"
                if failed_job
                else "remote_collection",
                "code": validation["code"]
                if validation
                else details.get(tid, "result_not_verified"),
                "task_id": tid,
                "job_id": job,
                "error": (
                    f"Slurm {outcome['state']}; exit_code={outcome.get('exit_code')}; "
                    if failed_job
                    else ""
                )
                + (
                    "Worker result rejected during reconciliation/scientific validation; no resubmission."
                    if validation
                    else "No verified worker result; task outcome is unconfirmed."
                ),
            }
        )
    diagnostic = {
        "schema_version": 1,
        "issues": issues,
        "jobs": job_rows,
        "collection_report": report,
        "unverified_task_ids": sorted(missing),
    }
    state.setdefault("relax_mc_remote", {})["failure_diagnostics"] = diagnostic
    return diagnostic


def compact_diagnostics(report, path):
    groups = {}
    for row in report.get("issues") or []:
        key = (row.get("origin"), row.get("code"), row.get("error"), row.get("job_id"))
        group = groups.setdefault(
            key, {k: row.get(k) for k in ("origin", "code", "error", "job_id", "traceback")}
        )
        group.setdefault("task_ids", []).append(row.get("task_id"))
    return {
        "counts": dict(Counter(r.get("origin") for r in report.get("issues") or [])),
        "verified_failed_task_ids": [
            r.get("task_id")
            for r in report.get("issues") or []
            if r.get("origin") == "remote_calculation"
            and r.get("code") in {"worker_failed", "worker_timeout"}
        ],
        "unverified_failure_task_ids": [
            r.get("task_id")
            for r in report.get("issues") or []
            if r.get("code") == "legacy_failure_unverified"
        ],
        "issue_groups": list(groups.values())[:20],
        "group_count": len(groups),
        "jobs": [
            {
                **{k: r.get(k) for k in ("job_id", "state", "exit_code", "reason", "query_error")},
                "logs": [
                    {"path": log.get("path"), "log_tail": str(log.get("log_tail") or "")[-2048:]}
                    for log in (r.get("logs") or [])[:3]
                ],
            }
            for r in report.get("jobs") or []
        ],
        "controller": report.get("controller"),
        "evidence_path": str(path),
        "unverified_task_ids": report.get("unverified_task_ids") or [],
    }


def read_return_diagnostics(directory, summary, tasks):
    directory = Path(directory)
    path = directory / "return_diagnostics.json"
    if path.is_file():
        if path.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("returned_diagnostics_too_large")
        report = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(report, dict) or report.get("schema_version") != 1:
            raise ValueError("returned_diagnostics_invalid_shape")
        for key in ("bundle_id", "plan_checksum", "model_version", "config_version"):
            if report.get(key) != summary.get(key):
                raise ValueError("returned_diagnostics_binding_mismatch:" + key)
        if not isinstance(report.get("issues", []), list) or any(
            not isinstance(r, dict) for r in report.get("issues", [])
        ):
            raise ValueError("returned_diagnostic_issues_invalid_shape")
    else:
        # Old immutable packages already carry worker errors and controller failure.json.
        from phase_agent.tools.remote.integrity import verify_result

        legacy_issues = []
        for task in tasks:
            if task.get("status") not in {"failed", "timeout"}:
                continue
            result = Path(task.get("result_path") or "missing/result.json")
            result = result if result.is_absolute() else directory / result
            result = result.with_name("result.json")
            verified = False
            if result.resolve().is_relative_to(directory.resolve()):
                try:
                    checked = verify_result(result, result.with_name("task.finished.json"), task)
                    verified = (
                        checked["valid"] and checked["result"].get("status") == task["status"]
                    )
                except OSError:
                    pass
            legacy_issues.append(
                {
                    "origin": "remote_calculation" if verified else "remote_collection",
                    "code": "worker_" + task["status"] if verified else "legacy_failure_unverified",
                    "task_id": task.get("task_id"),
                    "error": str(task.get("error") or task.get("failure_reason") or "")[:2048],
                }
            )
        report = {
            "issues": legacy_issues,
            "unverified_task_ids": [
                t["task_id"] for t in tasks if t.get("status") in {"pending", "running"}
            ],
        }
        path = directory / "failure.json"
        if path.is_file():
            failure = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(failure, dict):
                raise ValueError("returned_controller_diagnostic_invalid_shape")
            report["controller"] = {
                "origin": "remote_controller",
                "error": str(failure.get("reason") or "")[:2048],
                "traceback": str(failure.get("traceback") or "")[-4096:],
            }
        elif summary.get("reason"):
            path = directory / "result_summary.json"
            report["controller"] = {
                "origin": "remote_controller",
                "error": str(summary["reason"])[:2048],
            }
    for key in ("issues", "jobs"):
        if not isinstance(report.get(key, []), list) or any(
            not isinstance(row, dict) for row in report.get(key, [])
        ):
            raise ValueError("returned_diagnostics_invalid_" + key)
    for issue in report.get("issues") or []:
        for key in ("origin", "code", "error", "job_id", "task_id", "traceback"):
            if issue.get(key) is not None and not isinstance(issue[key], str):
                raise ValueError("returned_diagnostic_text_required:" + key)
        if issue.get("task_id") not in {task.get("task_id") for task in tasks}:
            raise ValueError("returned_diagnostic_task_outside_scope")
    if report.get("controller") is not None and not isinstance(report["controller"], dict):
        raise ValueError("returned_controller_diagnostic_invalid_shape")
    for job in report.get("jobs") or []:
        if not isinstance(job.get("logs", []), list) or any(
            not isinstance(row, dict) for row in job.get("logs", [])
        ):
            raise ValueError("returned_diagnostic_logs_invalid_shape")
    return compact_diagnostics(report, path)

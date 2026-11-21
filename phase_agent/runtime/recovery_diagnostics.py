"""Current return failure, independent of scientific facts and conversational memory."""

import json
from pathlib import Path, PurePosixPath
import zipfile

from phase_agent.tools.step_runner.file_protocol import read_json, write_json


def _path(state_path):
    path = Path(state_path)
    return path.with_name(path.name + ".recovery-error.json")


def _diagnostic(state_path):
    try:
        row = read_json(_path(state_path), {})
    except (OSError, ValueError, TypeError):
        return {"warning": "回收诊断记录无法读取"}
    if row == {} or row is None:
        return {}
    if not isinstance(row, dict) or not isinstance(row.get("error"), str):
        return {"warning": "回收诊断记录格式无效"}
    if row.get("state_path") != str(Path(state_path).resolve()):
        return {"warning": "回收诊断记录属于旧项目路径"}
    return row


def saved_recovery_error(state_path):
    return _diagnostic(state_path).get("error")


def recovery_diagnostic_notice(state_path):
    row = _diagnostic(state_path)
    if row.get("error"):
        from phase_agent.graphs.dialogue.decision import recovery_notice

        return recovery_notice(row["error"])
    if row.get("warning"):
        return row["warning"] + "；本次依据当前任务事实答复。合法回收重试可重新生成诊断。"
    return None


def record_recovery_error(state_path, error, fingerprint):
    write_json(
        _path(state_path),
        {
            "state_path": str(Path(state_path).resolve()),
            "error": str(error)[:500],
            "fingerprint": fingerprint,
        },
    )


def clear_recovery_error(state_path):
    _path(state_path).unlink(missing_ok=True)


_MAX_STATUS_BYTES = 65536
_MAX_LOG_BYTES = 8192


def _first_error(error):
    text = str(error or "No error message was saved")
    prefix = "remote_dispatch_failed:"
    if text.startswith(prefix):
        text = text[len(prefix):]
    if text.startswith("collection_incomplete:"):
        try:
            facts = json.loads(text.partition(":")[2])
            first = next(iter(facts.get("errors") or []), {})
            if first.get("error"):
                text = str(first["error"])
        except (ValueError, AttributeError, TypeError):
            pass
    return next((line.strip() for line in text.splitlines() if line.strip()), text)[:400]


def failure_summary(step, batch_id, error, resume_from):
    return {
        "failure_step": step,
        "batch_id": batch_id,
        "first_error": _first_error(error),
        "resume_from": resume_from,
    }


def _dispatch_failure_summary(batch_id, error):
    if "collection_incomplete:" in str(error)[:80]:
        return failure_summary("return_collection", batch_id, error, "verify_result_receipt")
    return failure_summary("dispatch", batch_id, error, "verify_scheduler_receipt")


def project_root(state_path):
    parent = Path(state_path).resolve().parent
    return parent.parent if parent.name == "workflow_state" else parent


def compact_recovery_failure(response, state_path=None):
    report = response.get("result_collection") or {}
    if report.get("recovered_count") or response.get("training_returns"):
        return response
    errors = report.get("transfer_errors") or []
    error = report.get("recovery_error") or response.get("error")
    if not errors and not error:
        return response
    first = errors[0] if errors else {}
    batch_id = first.get("batch_id") if isinstance(first, dict) else None
    archive = first.get("archive") if isinstance(first, dict) else None
    if not batch_id and archive and state_path:
        path = Path(archive).resolve()
        root = project_root(state_path)
        if path.parent.is_relative_to(root):
            batch_id = path.parent.relative_to(root).as_posix()
    if isinstance(first, dict):
        reason = first.get("reason") or error
    else:
        reason = first or error
    if str(reason).startswith("remote_dispatch_failed:"):
        return _dispatch_failure_summary(batch_id, reason)
    return failure_summary("return_import", batch_id, reason, "inspect_registered_return")


def _safe_project_path(root, value):
    path = Path(value).resolve()
    if not path.is_relative_to(root):
        raise ValueError("batch_diagnostic_path_outside_project")
    return path


def _batch_roots(state_path, root, batch_id):
    relative = PurePosixPath(batch_id.replace("\\", "/"))
    if len(relative.parts) > 1:
        if relative.parts[0] != "submissions" or ".." in relative.parts:
            raise ValueError("batch_diagnostic_requires_project_relative_submission_directory")
        return [_safe_project_path(root, root.joinpath(*relative.parts))]
    from phase_agent.persistence.state.field_index import IndexUnavailable, open_indexed_state

    state = open_indexed_state(state_path)
    message = "use the exact project-relative submissions directory"
    if state is None:
        raise ValueError(f"batch_diagnostic_index_unavailable; {message}")
    try:
        slurm = state.get("slurm_batches") or []
        mc = state.get("relax_mc_batches") or {}
        matches = []
        for row in slurm:
            if row.get("batch_id") == batch_id:
                handoff = row.get("handoff") or {}
                matches.extend((handoff.get("return_directory"), row.get("return_directory"),
                                row.get("results_directory")))
        if batch_id in mc:
            row = mc[batch_id]
            archive = row.get("expected_results_archive")
            matches.extend((Path(archive).parent if archive else None, row.get("return_directory")))
        state.verify()
    except IndexUnavailable:
        raise ValueError(f"batch_diagnostic_index_changed; {message}") from None
    if not any(matches):
        raise ValueError(f"batch_diagnostic_batch_not_found; {message}")
    return list(dict.fromkeys(_safe_project_path(root, value) for value in matches if value))


def _read_status(stream, limit=_MAX_STATUS_BYTES):
    data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("oversized_batch_diagnostic")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("invalid_batch_diagnostic")
    return value


def _batch_diagnostic(root, directory, include_details):
    zipped = _safe_project_path(root, directory / "diagnostics.zip")
    if zipped.is_file():
        with zipfile.ZipFile(zipped) as archive:
            names = set(archive.namelist())
            for member in ("dispatch-error.json", "run_status.json"):
                if member not in names:
                    continue
                with archive.open(member) as stream:
                    status = _read_status(stream)
                log = ""
                if include_details and "run.log" in names:
                    with archive.open("run.log") as stream:
                        log = stream.read(_MAX_LOG_BYTES).decode("utf-8", errors="replace")
                return member, status, log, zipped
    for member in ("dispatch-error.json", "run_status.json"):
        path = _safe_project_path(root, directory / member)
        if path.is_file():
            with path.open("rb") as stream:
                status = _read_status(stream)
            log = ""
            if include_details:
                log_path = _safe_project_path(root, directory / "run.log")
                if log_path.is_file():
                    with log_path.open("rb") as stream:
                        log = stream.read(_MAX_LOG_BYTES).decode("utf-8", errors="replace")
            return member, status, log, path
    return None


def _saved_transfer_error(state_path, roots):
    from phase_agent.persistence.state.field_index import IndexUnavailable, open_indexed_state

    state = open_indexed_state(state_path)
    if state is None:
        return None
    try:
        report = state.get("result_collection") or {}
        for key in ("transfer_errors", "historical_transfer_errors"):
            for row in report.get(key) or []:
                if isinstance(row, dict) and row.get("archive"):
                    if Path(row["archive"]).resolve().parent in roots:
                        state.verify()
                        return row
    except IndexUnavailable:
        return None
    return None


def batch_recovery_diagnostic(state_path, batch_id, *, include_details=False):
    """Read a batch's saved diagnostic without running collection."""
    root = project_root(state_path)
    roots = _batch_roots(state_path, root, batch_id)
    records = [row for directory in roots
               if (row := _batch_diagnostic(root, directory, include_details)) is not None]
    if not records:
        saved = _saved_transfer_error(state_path, roots)
        if saved is None:
            raise ValueError("batch_diagnostic_not_found")
        result = compact_recovery_failure(
            {"result_collection": {"transfer_errors": [{**saved, "batch_id": batch_id}]}},
            state_path,
        )
        if include_details:
            result["diagnostic"] = saved
        return result
    if len(records) > 1:
        raise ValueError("batch_diagnostic_ambiguous; use its exact submissions directory")
    member, status, log, path = records[0]
    error = status.get("error")
    code = status.get("returncode")
    if not error and (code not in (None, 0) or status.get("status") == "failed"):
        error = f"Controller exit code: {code}" if code is not None else "Controller reported failure"
    if not error:
        result = {"status": "no_failure_in_batch", "batch_id": batch_id}
    elif member == "dispatch-error.json":
        if status.get("code") == "slurm_qos_submit_limit":
            result = failure_summary("dispatch", batch_id, error, "retry_rejected_dispatch")
        else:
            result = _dispatch_failure_summary(batch_id, error)
    else:
        result = failure_summary("remote_export", batch_id, error, "inspect_batch_driver_log")
    if include_details:
        result.update(evidence_path=path.relative_to(root).as_posix(), diagnostic=status, log=log)
    return result

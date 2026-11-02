"""Observe existing returns under their bound configuration before any decision."""

from copy import deepcopy
import hashlib
import json
from threading import RLock
from phase_agent.runtime.turn_process import timed_call
from phase_agent.interface_compatibility import call_with_optimizations
from pathlib import Path

from phase_agent.configuration.runtime.build_effective_run_config import (
    build_effective_run_config,
)
from phase_agent.configuration.session.phase_cache_location import phase_cache_location
from phase_agent.tools.remote.transfer_bundle import has_registered_return
from phase_agent.tools.state.task_waiting import awaiting_task_result
from phase_agent.tools.state.reconcile_task_results import reconcile_task_results
from phase_agent.tools.workflows.apply_scientific_feedback import (
    apply_scientific_feedback,
)
from phase_agent.analysis.phase.update_local_mlip_hull_pool import (
    update_local_mlip_hull_pool,
)
from phase_agent.tools.workflows.lifecycle_support import (
    _save_runner_state,
    _dft_comparison_evaluator,
)


_RECOVERY_CACHE = {}
_RECOVERY_CACHE_LOCK = RLock()


def _training_return_paths(job):
    handoff = job.get("handoff") or {}
    roots = {handoff.get("return_directory"), handoff.get("results_directory")}
    if job.get("directory"):
        roots.update({job["directory"], str(Path(job["directory"]) / "results")})
    return sorted(str(root) for root in roots if root)


def _training_return_facts(state):
    from phase_agent.tools.remote.transfer_bundle import _registered_archives

    facts = []
    for key, job in sorted((state.get("remote_finetune_jobs") or {}).items()):
        artifacts = set()
        for value in _training_return_paths(job):
            root = Path(value)
            paths = [
                *_registered_archives(root),
                root / "training.finished.json",
                root / "models.json",
                root / "remote_artifacts.json",
            ]
            for path in paths:
                try:
                    if path.is_file():
                        stat = path.stat()
                        artifacts.add((str(path), stat.st_size, stat.st_mtime_ns))
                except OSError:
                    continue
        if artifacts:
            facts.append({"job_id": key, "artifacts": sorted(artifacts)})
    return facts


def recovery_candidate_fingerprint(state, *, legacy_training_metadata=False):
    """Return a cheap identity for the currently registered return sources."""
    from phase_agent.tools.remote.transfer_bundle import (
        _registered_archives,
        dispatch_failure,
        _pending_return_state,
    )

    state = _pending_return_state(state)

    # Prepared input batches can add many slurm rows without adding a return.
    # Fingerprint only actual result archives/dispatch diagnostics; otherwise a
    # new DFT input batch reopens an already acknowledged old recovery gate.
    slurm_returns = []
    for row in state.get("slurm_batches") or []:
        handoff = row.get("handoff") or {}
        roots = {
            row.get("results_directory"),
            row.get("return_directory"),
            row.get("return_archive"),
            handoff.get("results_directory"),
            handoff.get("return_directory"),
            handoff.get("return_archive"),
        }
        artifacts = []
        for value in roots:
            if not value:
                continue
            root = Path(value)
            if root.is_file() and root.suffix == ".zip":
                candidates = [root]
            elif root.is_dir():
                candidates = _registered_archives(root)
            else:
                candidates = []
            for archive in candidates:
                try:
                    stat = archive.stat()
                except OSError:
                    continue
                artifacts.append((str(archive), stat.st_size, stat.st_mtime_ns))
        diagnostics = None
        for value in (handoff.get("return_directory"), row.get("return_directory")):
            if value:
                failure = dispatch_failure(Path(value))
                if failure:
                    diagnostics = (str(Path(value) / "diagnostics.zip"), failure)
                    break
        if artifacts or diagnostics:
            slurm_returns.append(
                {
                    "batch_id": row.get("batch_id"),
                    "artifacts": sorted(set(artifacts)),
                    "diagnostic": diagnostics,
                }
            )

    task_returns = []
    for row in state.get("tasks") or []:
        if not awaiting_task_result(row):
            continue
        value = row.get("result_path")
        if not isinstance(value, str):
            continue
        path = Path(value)
        if not path.is_file():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        task_returns.append(
            {
                "task_id": row.get("task_id"),
                "path": str(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    payload = {
        "relax_mc": [
            {
                "bundle_id": key,
                "archive": row.get("expected_results_archive"),
                "import_checksum": row.get("import_checksum"),
                "status": row.get("status"),
                "result_summary": row.get("result_summary"),
            }
            for key, row in sorted((state.get("relax_mc_batches") or {}).items())
        ],
        "slurm": sorted(slurm_returns, key=lambda item: str(item.get("batch_id") or "")),
        "task_returns": sorted(task_returns, key=lambda item: str(item.get("task_id") or "")),
        "remote_finetune": [
            {
                "job_id": key,
                "directory": row.get("directory"),
                "status": row.get("status"),
                "activated": row.get("activated"),
            }
            for key, row in sorted((state.get("remote_finetune_jobs") or {}).items())
        ]
        if legacy_training_metadata
        else _training_return_facts(state),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


def _legacy_acknowledgement_is_safe(state, *, backlog, returned=False):
    """Accept pre-input-preparation recovery acks without reopening old returns."""
    if backlog:
        return False
    # A legacy acknowledgement has no binding to a newly observed return
    # archive.  Exact fingerprints are handled by the caller; any mismatched
    # current return must go through the normal explicit recovery gate.
    if returned:
        return False
    ack = state.get("recovery_acknowledgement") or {}
    if ack.get("scope") != "explicit_user_recovery_approval":
        return False
    report = state.get("execution_recovery_report") or {}
    if report.get("status") != "clear" or report.get("unsettled"):
        return False
    return not any(
        awaiting_task_result(row)
        and isinstance(row.get("result_path"), str)
        and Path(row["result_path"]).is_file()
        for row in state.get("tasks") or []
    )


def _return_watch_signature(state):
    from stat import S_ISDIR

    paths = set()
    for task in state.get("tasks") or []:
        if awaiting_task_result(task) and isinstance(task.get("result_path"), str):
            paths.add(task["result_path"])
    for row in state.get("slurm_batches") or []:
        handoff = row.get("handoff") or {}
        for key in ("return_directory", "return_archive", "results_directory"):
            value = handoff.get(key) or row.get(key)
            if value:
                paths.add(str(value))
    for row in (state.get("relax_mc_batches") or {}).values():
        if row.get("expected_results_archive"):
            paths.add(str(row["expected_results_archive"]))
        if row.get("return_directory"):
            paths.add(str(row["return_directory"]))
    for job in (state.get("remote_finetune_jobs") or {}).values():
        for root in _training_return_paths(job):
            paths.add(root)
            paths.update(
                str(Path(root) / name)
                for name in ("training.finished.json", "models.json", "remote_artifacts.json")
            )
    signature = []
    for value in sorted(paths):
        path = Path(value)
        try:
            stat = path.stat()
            signature.append((value, stat.st_mtime_ns, stat.st_size))
            if S_ISDIR(stat.st_mode):
                # Directory timestamps can lag new files on Windows. Inspect
                # only immediate archives, without opening or walking them.
                for archive in sorted(path.glob("*.zip")):
                    try:
                        archive_stat = archive.stat()
                    except OSError:
                        continue
                    signature.append((str(archive), archive_stat.st_mtime_ns, archive_stat.st_size))
        except OSError:
            signature.append((value, None, None))
    return tuple(signature)


def _recovery_cache_key(state, state_path):
    path = Path(state_path).resolve()
    try:
        stat = path.stat()
        state_file = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        state_file = (None, None)
    pending_tasks = tuple(
        (row.get("task_id"), row.get("status"), row.get("result_path"))
        for row in state.get("tasks") or []
        if awaiting_task_result(row)
    )
    return (
        str(path),
        state_file,
        state.get("active_model_version"),
        state.get("confirmed_config_version"),
        (state.get("recovery_acknowledgement") or {}).get("fingerprint"),
        tuple(sorted(state.get("processed_task_ids") or [])),
        tuple(sorted(state.get("feedback_processed_task_ids") or [])),
        pending_tasks,
        _return_watch_signature(state),
    )


def recovery_candidate(state, *, state_path=None):
    """Read-only detection of returns that still need explicit approval."""
    cache_key = _recovery_cache_key(state, state_path) if state_path else None
    if cache_key is not None:
        with _RECOVERY_CACHE_LOCK:
            cached = _RECOVERY_CACHE.get(cache_key)
        if cached is not None:
            return deepcopy(cached)
    processed = set(state.get("processed_task_ids") or [])
    feedback_processed = set(state.get("feedback_processed_task_ids") or [])
    backlog = [
        row
        for row in state.get("tasks") or []
        if row.get("task_id") in processed - feedback_processed
        and row.get("stage")
        in {"relax_and_feature", "deep_search", "dft_single_point", "dft_relax"}
        and row.get("status") in {"completed", "failed", "timeout", "cancelled"}
    ]
    returned = has_registered_return(state) or any(
        awaiting_task_result(row)
        and isinstance(row.get("result_path"), str)
        and Path(row["result_path"]).is_file()
        for row in state.get("tasks") or []
    )
    fingerprint = recovery_candidate_fingerprint(state)
    acknowledged = state.get("recovery_acknowledgement") or {}
    legacy_training_ack = False
    if (
        returned
        and not backlog
        and acknowledged.get("scope") == "explicit_user_recovery_approval"
        and acknowledged.get("fingerprint") != fingerprint
    ):
        # Upgrade acknowledgements written before prepared-job metadata was
        # removed. Never use this compatibility path to hide an active return.
        active_returns = any(
            not ((state.get("remote_finetune_jobs") or {}).get(row["job_id"]) or {}).get(
                "activated"
            )
            for row in _training_return_facts(state)
        )
        legacy_training_ack = not active_returns and acknowledged.get("fingerprint") == (
            recovery_candidate_fingerprint(state, legacy_training_metadata=True)
        )
    already_reviewed = (
        bool(returned)
        and not backlog
        and (
            acknowledged.get("fingerprint") == fingerprint
            or legacy_training_ack
            or _legacy_acknowledgement_is_safe(state, backlog=backlog, returned=returned)
        )
    )
    result = {
        "returned": bool(returned),
        "backlog_count": len(backlog),
        "fingerprint": fingerprint,
        "pending": bool((returned or backlog) and not already_reviewed),
    }
    if cache_key is not None:
        with _RECOVERY_CACHE_LOCK:
            _RECOVERY_CACHE.clear()
            _RECOVERY_CACHE[cache_key] = deepcopy(result)
    return result


def recover_saved_results(state, **dependencies):
    """Collect and analyze evidence only; never migrate, propose or dispatch work."""
    collector = dependencies.get("task_runner") or dependencies.get("result_collector")
    manager = dependencies.get("manager")
    bound = state.get("confirmed_config")
    version = state.get("confirmed_config_version")
    if collector is None or manager is None or not bound or not version:
        return state
    candidate = recovery_candidate(state, state_path=dependencies.get("state_path"))
    if not candidate["pending"]:
        return state
    processed = set(state.get("processed_task_ids") or [])
    feedback_processed = set(state.get("feedback_processed_task_ids") or [])
    backlog = [
        row
        for row in state.get("tasks") or []
        if row.get("task_id") in processed - feedback_processed
        and row.get("stage")
        in {"relax_and_feature", "deep_search", "dft_single_point", "dft_relax"}
        and row.get("status") in {"completed", "failed", "timeout", "cancelled"}
    ]
    session = {
        "status": "confirmed",
        "confirmed_snapshot": {
            "config_version": version,
            "config": deepcopy(bound),
        },
    }
    config = build_effective_run_config(session, dependencies.get("run_config"))
    if state.get("active_model"):
        config["mlip"] = {
            **(config.get("mlip") or {}),
            **deepcopy(state["active_model"]),
        }
    evaluator = _dft_comparison_evaluator(dependencies, config, state)
    if hasattr(collector, "collect_results_with_report"):
        collected = timed_call(
            "saved_return_collection",
            collector.collect_results_with_report,
            state,
            category="processing",
        )
        results, report = collected["results"], collected["report"]
    else:
        results = collector.collect_results(state)
        report = {"recovered_count": len(results)}
    incoming_ids = {row.get("task_id") for row in results}
    results = [
        *results,
        *(row for row in backlog if row.get("task_id") not in incoming_ids),
    ]
    current = deepcopy(state)
    report = _classify_transfer_diagnostics(report, state)
    current["result_collection"] = deepcopy(report)
    if not results:
        # A soft collector failure is not successful observation. The results
        # already include saved feedback backlog, so valid partial evidence is
        # never blocked here. Training returns have their own validation graph.
        failure = _empty_collection_failure(report, state)
        if failure:
            current["result_collection"]["recovery_error"] = failure
            return current
        return _complete_saved_comparisons(current, evaluator, manager, config, dependencies)
    reconciled = call_with_optimizations(
        reconcile_task_results, current, results, _optimizations={"_owned": True}
    )
    state_path = dependencies.get("state_path") or config.get("state_path")
    # Persist verified completion and budget settlement before expensive phase
    # recognition. A restart resumes outstanding feedback from these records.
    _save_runner_state(reconciled["state"], state_path)
    feedback = timed_call(
        "saved_scientific_feedback",
        call_with_optimizations,
        apply_scientific_feedback,
        reconciled["state"],
        results,
        category="processing",
        manager=manager,
        ledger_path=config.get("ledger_path"),
        phase_diagram_directory=config.get("phase_diagram_directory"),
        final_frame_mlip_evaluator=evaluator,
        active_model_version=(config.get("mlip") or {}).get("version")
        or (config.get("mlip") or {}).get("name"),
        phase_references=dependencies.get("phase_references"),
        _optimizations={"_owned": True},
        phase_identification_cache_path=dependencies.get("phase_identification_cache_path")
        or (phase_cache_location(state_path) if state_path else None),
    )
    current = call_with_optimizations(
        update_local_mlip_hull_pool,
        feedback["state"],
        config=config,
        path=config.get("branch_energy_pool_ledger_path"),
        _optimizations={"_owned": True},
    )
    current["result_collection"]["feedback_rejected"] = feedback.get("rejected") or []
    current = _complete_saved_comparisons(current, evaluator, manager, config, dependencies)
    _save_runner_state(current, state_path)
    return current


def _complete_saved_comparisons(state, evaluator, manager, config, dependencies):
    """Repair verified saved pairs without recollecting labels or settling costs."""
    completed = {
        row.get("task_id")
        for row in state.get("dft_mlip_comparisons") or []
        if row.get("status") == "completed"
    }
    ids = [
        row["task_id"]
        for row in state.get("dft_dataset_records") or []
        if row.get("task_id") not in completed
        and row.get("status") == "completed"
        and row.get("converged") is True
        and row.get("checks_passed") is True
        and row.get("training_ready") is True
    ]
    if not ids:
        return state
    from phase_agent.tools.workflows.refresh_dft_comparisons import refresh_dft_comparisons

    updated = refresh_dft_comparisons(
        state, assessment=None, evaluator=evaluator, manager=manager, task_ids=ids
    )
    if updated.get("dft_mlip_comparisons") != state.get("dft_mlip_comparisons"):
        from phase_agent.analysis.feedback.dft_result_products import export_dft_products

        export_dft_products(updated, config.get("phase_diagram_directory"))
        _save_runner_state(updated, dependencies.get("state_path") or config.get("state_path"))
    return updated


def _empty_collection_failure(report, state):
    failures = {
        key: report[key]
        for key in (
            "invalid_count",
            "missing_structure_count",
            "missing_marker_count",
            "transfer_errors",
        )
        if report.get(key)
    }
    if not failures:
        return None
    jobs = state.get("remote_finetune_jobs") or {}
    pending_training = any(
        not (job := jobs[row["job_id"]]).get("activated")
        and job.get("status") not in {"abandoned", "cancelled", "superseded"}
        and job.get("original_model_version") == state.get("active_model_version")
        for row in _training_return_facts(state)
    )
    if pending_training:
        return None
    return "没有通过本次回收核验的有效结果：" + str(failures)[:350]


def _classify_transfer_diagnostics(report, state):
    """Retain dispatch diagnostics without blocking already reconciled tasks."""
    if not report.get("transfer_errors"):
        return report
    from phase_agent.tools.remote.legacy_recovery import recovery_handoffs

    scopes = {}
    for batch in state.get("slurm_batches") or []:
        folder = (batch.get("handoff") or {}).get("return_directory")
        if folder:
            scopes.setdefault(str(Path(folder).resolve()), set()).update(
                batch.get("task_ids") or []
            )
    for handoff in recovery_handoffs(state):
        scopes.setdefault(str(Path(handoff["return_directory"]).resolve()), set()).update(
            handoff.get("task_ids") or []
        )
    processed = set(state.get("processed_task_ids") or [])
    tasks = {row.get("task_id"): row for row in state.get("tasks") or []}
    current, historical = [], []
    for error in report["transfer_errors"]:
        archive = Path(error.get("archive") or "") if isinstance(error, dict) else None
        ids = scopes.get(str(archive.parent.resolve()), set()) if archive else set()
        if not ids and isinstance(error, dict):
            # Older diagnostics may outlive their handoff metadata. Recover
            # the task identity from the bounded dispatch error text so a
            # settled task cannot keep reopening the recovery gate.
            import re

            reason = str(error.get("reason", ""))
            ids = set(re.findall(r'"task_id"\s*:\s*"([^"]+)"', reason))
        settled = (
            bool(ids)
            and ids.issubset(processed)
            and all(
                tasks.get(tid, {}).get("status") in {"completed", "failed", "timeout", "cancelled"}
                for tid in ids
            )
        )
        if (
            settled
            and archive.name == "diagnostics.zip"
            and str(error.get("reason", "")).startswith("remote_dispatch_failed:")
        ):
            historical.append(error)
        else:
            current.append(error)
    return {**report, "transfer_errors": current, "historical_transfer_errors": historical}

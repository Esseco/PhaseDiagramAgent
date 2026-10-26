"""Observe existing returns under their bound configuration before any decision."""

from copy import deepcopy
import hashlib
import json
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


def recovery_candidate_fingerprint(state):
    """Return a cheap identity for the currently registered return sources."""
    from phase_agent.tools.remote.transfer_bundle import (
        _registered_archives,
        dispatch_failure,
    )

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
            {"task_id": row.get("task_id"), "path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
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
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


def _legacy_acknowledgement_is_safe(state, *, backlog):
    """Accept pre-input-preparation recovery acks without reopening old returns."""
    if backlog:
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


def recovery_candidate(state):
    """Read-only detection of returns that still need explicit approval."""
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
    already_reviewed = bool(returned) and not backlog and (
        acknowledged.get("fingerprint") == fingerprint
        or _legacy_acknowledgement_is_safe(state, backlog=backlog)
    )
    return {
        "returned": bool(returned),
        "backlog_count": len(backlog),
        "fingerprint": fingerprint,
        "pending": bool((returned or backlog) and not already_reviewed),
    }


def recover_saved_results(state, **dependencies):
    """Collect and analyze evidence only; never migrate, propose or dispatch work."""
    collector = dependencies.get("task_runner") or dependencies.get("result_collector")
    manager = dependencies.get("manager")
    bound = state.get("confirmed_config")
    version = state.get("confirmed_config_version")
    if collector is None or manager is None or not bound or not version:
        return state
    candidate = recovery_candidate(state)
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
    current["result_collection"] = deepcopy(report)
    if not results:
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

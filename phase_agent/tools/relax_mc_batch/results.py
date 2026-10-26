"""Verify, inspect and merge batch returns through existing ledger/reconciliation APIs."""

from copy import deepcopy
from pathlib import Path
import json
import zipfile

from phase_agent.tools.remote.transfer_bundle import (
    import_bundle,
    checksum,
    validate_bundle,
    local_path,
    check_return_paths,
)
from phase_agent.tools.relax_mc_batch.files import read, rebase, PATH_KEYS
from phase_agent.tools.relax_mc_batch.handoff import result_archive, output_directory
from phase_agent.tools.relax_mc_batch.failure_diagnostics import read_return_diagnostics
from phase_agent.tools.state.reconcile_task_results import reconcile_task_results
from phase_agent.tools.workflows.apply_scientific_feedback import (
    apply_scientific_feedback,
)
from phase_agent.tools.budget.reserve_budget import (
    reserve_owned_budget,
    reserve_owned_budget as _reserve_returned_budget,
)


def inspect_archive(path):
    # Stream full integrity checks; inspection never materializes thousands of files.
    validate_bundle(path, extra_result_suffixes=(".csv", ".bin"))
    with zipfile.ZipFile(path) as archive:
        return {
            "summary": json.loads(archive.read("result_summary.json")),
            "candidate_pool": json.loads(archive.read("candidate_pool.json")),
        }


def _scientific_values(value):
    # Raw worker JSON retains remote paths; portable State rebases them locally.
    # Compare the scientific content without falsely rejecting that relocation.
    if isinstance(value, dict):
        return {
            key: _scientific_values(item) for key, item in value.items() if key not in PATH_KEYS
        }
    if isinstance(value, list):
        return [_scientific_values(item) for item in value]
    return value


def _restore_missing_return_files(archive, destination):
    """Re-extract missing files from the same verified archive; never replace conflicts."""
    with zipfile.ZipFile(archive) as zipped:
        rows = json.loads(zipped.read("transfer_manifest.json"))["files"]
    root = Path(destination).resolve()
    missing = 0
    for row in rows:
        target = (root / row["path"]).resolve()
        if not target.is_relative_to(root):
            raise ValueError("returned_artifact_outside_destination")
        missing += not target.is_file()
    if missing:
        import_bundle(archive, destination, extra_result_suffixes=(".csv", ".bin"))
    return missing


def import_results(*, action, context, _owned=False):
    state = context.get("event_state") or {}
    bundle_id = (action.get("parameters") or {}).get("bundle_id")
    record = (state.get("relax_mc_batches") or {}).get(bundle_id)
    if not record:
        raise ValueError("unknown_registered_relax_mc_bundle")
    if record.get("status") in {"cancelled", "superseded"}:
        raise ValueError("cancelled_relax_mc_bundle_requires_manual_result_review")
    config = context["effective_config"]
    model = (
        state.get("active_model_version")
        or config["mlip"].get("version")
        or config["mlip"].get("name")
    )
    if model != record["model_version"]:
        raise ValueError(
            "batch_model_or_config_version_mismatch; inspect historical results without activating them"
        )
    archive = result_archive(record)
    from phase_agent.tools.relax_mc_batch.recovery_receipt import (
        selected_recovery,
        validate_recovery_scope,
    )

    recovery = selected_recovery(record)
    if context["config_version"] != record["config_version"] and recovery is None:
        raise ValueError(
            "batch_model_or_config_version_mismatch; inspect historical results without activating them"
        )
    if recovery is not None:
        validate_recovery_scope(archive, record, state)
    digest = checksum(archive)
    if record.get("import_checksum") == digest:
        current = state if _owned else deepcopy(state)
        prior = current["relax_mc_batches"][bundle_id]
        destination = prior.get("return_directory")
        if destination:
            restored = _restore_missing_return_files(archive, destination)
            returned = read(Path(destination) / "workflow_state.json")
            prior["remote_diagnostics"] = read_return_diagnostics(
                destination, prior.get("result_summary") or {}, returned.get("tasks") or []
            )
            if restored:
                prior["local_recovery"] = {
                    "origin": "local_import",
                    "code": "missing_verified_artifacts_restored",
                    "restored_file_count": restored,
                    "submitted": False,
                }
            prior.pop("return_error", None)
            prior.pop("return_diagnostic", None)
        from phase_agent.analysis.phase.phase_round_analysis import publish_relax_mc_analysis

        # Pre-feature historical imports have no round-bound analysis. Do not
        # label today's accumulated hull as an old round on a routine rescan.
        if prior.get("phase_analysis") is not None:
            prior["phase_analysis"] = publish_relax_mc_analysis(
                current, bundle_id, directory=config.get("phase_diagram_directory")
            )
        return {
            "status": "already_imported",
            "state": current,
            "submitted": False,
        }
    cancelled = [
        task["task_id"]
        for task in state.get("tasks") or []
        if task.get("relax_mc_bundle_id") == bundle_id and task.get("recovery_cancelled") is True
    ]
    if cancelled:
        current = state if _owned else deepcopy(state)
        current["relax_mc_batches"][bundle_id]["return_diagnostic"] = {
            "origin": "local_import",
            "stage": "cancelled_wait_review",
            "code": "cancelled_mc_return_requires_manual_review",
            "error": "本批部分MC已明确停止本地等待；新回传保留原包，需核对后人工接纳，不自动激活结果或结算费用。",
            "task_ids": cancelled,
            "archive_checksum": digest,
            "evidence_path": str(archive),
        }
        return {"status": "manual_review_required", "state": current, "submitted": False}
    # A changed return is transactional even for an owned caller: failed
    # validation must not leave partial task/budget updates in its input.
    state = deepcopy(state)
    record = state["relax_mc_batches"][bundle_id]
    # Preserve extended paths through receipt checks and final-geometry reads,
    # not only ZIP staging: HPC member names can exceed Windows' legacy limit.
    from phase_agent.tools.remote.model_upload_directory import model_upload_directory

    from phase_agent.tools.remote.submission_architecture import epoch_evidence_directory

    epoch = model_upload_directory(
        config["upload_batches_directory"], state, record["model_version"]
    )
    destination = local_path(epoch_evidence_directory(epoch, "verified_results") / digest[:16])
    path_report = check_return_paths(archive, destination)
    import_bundle(archive, destination, extra_result_suffixes=(".csv", ".bin"))
    summary = read(destination / "result_summary.json")
    for key, expected in (
        ("bundle_id", bundle_id),
        ("plan_checksum", record["plan_checksum"]),
        ("model_version", record["model_version"]),
        ("config_version", record["config_version"]),
    ):
        if summary.get(key) != expected:
            raise ValueError("returned_batch_binding_mismatch:" + key)
    returned = rebase(read(destination / "workflow_state.json"), destination)
    returned_tasks = returned.get("tasks") or []
    remote_diagnostics = read_return_diagnostics(destination, summary, returned_tasks)
    if len({r.get("task_key") for r in returned_tasks}) != len(returned_tasks):
        raise ValueError("duplicate_returned_task_keys")
    from phase_agent.tools.remote.integrity import verify_result

    for task in returned_tasks:
        if task.get("branch_id") not in record["branch_ids"] or task.get("model_version") != model:
            raise ValueError("returned_task_outside_approved_scope")
        if task.get("status") not in {"completed", "failed", "timeout", "cancelled"}:
            continue
        path = Path(task.get("result_path") or "")
        if path.name != "result.json":
            path = path.parent / "result.json" if path.suffix else path / "result.json"
        if task.get("status") == "completed":
            expected = {
                key: task.get(key)
                for key in (
                    "task_id",
                    "task_key",
                    "batch_id",
                    "config_version",
                    "model_version",
                    "task_checksum",
                    "protocol_version",
                    "input_file_version",
                )
            }
            verified = verify_result(path, path.with_name("task.finished.json"), expected)
            if not verified["valid"]:
                raise ValueError("returned_task_integrity_failed:" + verified["reason"])
            raw = read(path)
            for key in (
                "energy",
                "energy_per_atom",
                "energy_unit",
                "composition",
                "qbc",
                "pool_summary",
            ):
                if _scientific_values((raw.get("outputs") or {}).get(key)) != _scientific_values(
                    (task.get("outputs") or {}).get(key)
                ):
                    raise ValueError("returned_state_result_disagreement:" + key)
    existing = {t["task_id"]: t for t in state.get("tasks") or []}
    for task in returned_tasks:
        sid = task.get("structure_id")
        if sid not in context["manager"].data["structures"]:
            raise ValueError("returned_structure_not_registered")
        if context["manager"].data["structures"][sid]["branch_id"] != task["branch_id"]:
            raise ValueError("returned_structure_branch_mismatch")
        if task["task_id"] in existing and any(
            existing[task["task_id"]].get(k) != task.get(k)
            for k in ("task_key", "structure_id", "stage")
        ):
            raise ValueError("returned_registered_task_mismatch")
        if task["task_id"] not in existing:
            if task.get("stage") != "deep_search" or task.get("segment_index") not in (
                0,
                1,
                2,
            ):
                raise ValueError("unexpected_remote_task")
            state.setdefault("tasks", []).append({**deepcopy(task), "status": "pending"})
            reserved = reserve_owned_budget(
                state,
                task_key=task["task_key"],
                stage=task["stage"],
                amount=task["planned_relative_cost"],
                limits=config["budgets"],
                config_version=record["config_version"],
                model_version=model,
            )
            if reserved["status"] == "rejected":
                raise ValueError("returned_cost_reservation_requires_review")
            state = reserved["state"]
    terminal = [
        t
        for t in returned_tasks
        if t.get("status") in {"completed", "failed", "timeout", "cancelled"}
    ]
    state = reconcile_task_results(state, terminal)["state"]
    state = apply_scientific_feedback(
        state,
        terminal,
        manager=context["manager"],
        ledger_path=config.get("ledger_path"),
        phase_diagram_directory=config.get("phase_diagram_directory"),
        active_model_version=model,
        phase_references=context.get("phase_references"),
    )["state"]
    old = state.setdefault("tiered_mc_state", {}).setdefault("segments", [])
    known = {r.get("task_key") for r in old}
    old.extend(
        deepcopy(t)
        for t in returned_tasks
        if t.get("stage") == "deep_search" and t["task_key"] not in known
    )
    allocations = state.setdefault("mc_second_round_allocations", [])
    for row in returned.get("mc_second_round_allocations") or []:
        if row not in allocations:
            allocations.append(deepcopy(row))
    if returned.get("mc_second_round_allocation"):
        state["mc_second_round_allocation"] = deepcopy(returned["mc_second_round_allocation"])
    candidate = read(destination / "candidate_pool.json")
    candidate = rebase(candidate, destination)
    state.setdefault("relax_mc_returned_candidates", {})[bundle_id] = candidate
    # Standard decision context already exposes qbc_candidates; retain other active candidates.
    by_id = {r.get("candidate_id"): r for r in state.get("qbc_candidates") or []}
    by_id.update({r["candidate_id"]: r for r in candidate["branch_candidates"]})
    state["qbc_candidates"] = list(by_id.values())
    state["relax_mc_batches"][bundle_id].update(
        status=summary["status"],
        import_checksum=digest,
        result_summary=summary,
        return_directory=str(destination),
        return_path_check=path_report,
        remote_diagnostics=remote_diagnostics,
    )
    state["relax_mc_batches"][bundle_id].pop("return_error", None)
    state["relax_mc_batches"][bundle_id].pop("return_diagnostic", None)
    from phase_agent.analysis.phase.phase_quality_actions import automatic_round_review

    try:
        ledger_path = config.get("ledger_path")
        if not ledger_path:
            raise ValueError("缺少台账位置，不能创建排除备份；本轮保留并交人工核查")
        backup_root = Path(ledger_path).resolve().parents[2] / "history_backups" / "phase_quality"
        checked = automatic_round_review(
            state,
            context["manager"],
            bundle_id=bundle_id,
            backup_root=backup_root,
            directory=config.get("phase_diagram_directory"),
        )
        state = checked["state"]
        if checked.get("excluded_record_ids"):
            context["manager"].save(ledger_path)
        state["relax_mc_batches"][bundle_id]["phase_quality_action"] = {
            k: v for k, v in checked.items() if k != "state"
        }
    except (OSError, ValueError, TypeError, KeyError) as error:
        state["relax_mc_batches"][bundle_id]["phase_quality_action"] = {
            "status": "manual_review_required",
            "reason": str(error),
            "submitted": False,
        }
    from phase_agent.analysis.phase.phase_round_analysis import publish_relax_mc_analysis

    state["relax_mc_batches"][bundle_id]["phase_analysis"] = publish_relax_mc_analysis(
        state, bundle_id, directory=config.get("phase_diagram_directory")
    )
    from phase_agent.analysis.phase.phase_quality_reports import candidate_quality_evidence

    state["qbc_candidates"] = candidate_quality_evidence(state, state.get("qbc_candidates") or [])
    if summary.get("recovery"):
        state.setdefault("result_recoveries", {})[bundle_id] = {
            **deepcopy(summary["recovery"]),
            "status": summary["status"],
            "archive_checksum": digest,
            "archive": str(archive),
        }
    return {
        "status": "completed",
        "state": state,
        "result_summary": summary,
        "candidate_pool": candidate,
        "submitted": False,
    }


def _driver_diagnostic(record):
    output = (
        output_directory(record["directory"])
        if record.get("directory")
        else result_archive(record).parent
    )
    status_file, diagnostic = output / "run_status.json", output / "diagnostics.zip"
    if not status_file.is_file() and not diagnostic.is_file():
        return None
    log_tail = ""
    if status_file.is_file():
        if status_file.stat().st_size > 65536:
            raise ValueError("oversized_diagnostic_status")
        status = read(status_file)
    else:
        with zipfile.ZipFile(diagnostic) as archive:
            if archive.getinfo("run_status.json").file_size > 65536:
                raise ValueError("oversized_diagnostic_status")
            status = json.loads(archive.read("run_status.json"))
    if not isinstance(status, dict):
        raise ValueError("remote_status_object_required")
    if diagnostic.is_file():
        with zipfile.ZipFile(diagnostic) as archive:
            if "run.log" in archive.namelist() and archive.getinfo("run.log").file_size <= 65536:
                log_tail = archive.read("run.log").decode("utf-8", "replace")[-4096:]
    failed = bool(status.get("error")) or status.get("returncode") not in (0, None)
    return {
        "origin": "remote_controller" if failed else "return_transfer",
        "code": "remote_driver_failed" if failed else "remote_driver_status",
        "stage": "remote_export",
        "exit_code": status.get("returncode"),
        "error": status.get("error"),
        "log_tail": log_tail,
        "evidence_path": str(diagnostic if diagnostic.is_file() else status_file),
    }


def recover_available(state, context, *, _owned=False):
    current = state if _owned else deepcopy(state)
    for bid, record in list((current.get("relax_mc_batches") or {}).items()):
        record = current["relax_mc_batches"][bid]
        if record.get("status") in {"cancelled", "superseded"}:
            continue
        try:
            archive_exists = result_archive(record).is_file()
        except (OSError, ValueError, KeyError) as error:
            record["return_error"] = str(error)
            record["return_diagnostic"] = {
                "origin": "local_import",
                "code": "invalid_recovery_receipt",
                "error": str(error),
            }
            continue
        try:
            driver = _driver_diagnostic(record)
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
            driver = {
                "origin": "local_import",
                "stage": "diagnostic_read",
                "code": type(error).__name__,
                "error": str(error)[:2048],
            }
        if driver:
            record["driver_diagnostic"] = driver
        if not archive_exists:
            if driver:
                record["return_error"] = (
                    "回传文件夹缺少 results.zip；控制/导出退出码 "
                    + str(driver.get("exit_code"))
                    + "；"
                    + str(driver.get("error") or "请核对回传清单与日志")
                )
                record["return_diagnostic"] = driver
            continue
        try:
            config = context.get("effective_config") or {}
            mlip = config.get("mlip") or {}
            model = current.get("active_model_version") or mlip.get("version") or mlip.get("name")
            historical = model != record.get("model_version") or context.get(
                "config_version"
            ) != record.get("config_version")
            # A known historical receipt is not a request to activate or re-import it.
            if historical and record.get("import_checksum") == checksum(result_archive(record)):
                diagnostic = record.get("return_diagnostic") or {}
                if "batch_model_or_config_version_mismatch" in str(
                    diagnostic.get("error") or record.get("return_error") or ""
                ):
                    record.pop("return_error", None)
                    record.pop("return_diagnostic", None)
                continue
            from phase_agent.interface_compatibility import call_with_optimizations

            result = call_with_optimizations(
                import_results,
                _optimizations={"_owned": True},
                action={"parameters": {"bundle_id": bid}},
                context={**context, "event_state": current},
            )
            current = result["state"]
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
            detail = str(error)
            if isinstance(error, OSError):
                from phase_agent.tools.remote.transfer_bundle import _path_length

                failed_path = error.filename or "未提供"
                detail = (
                    f"阶段=本地回收文件读写；本地路径={failed_path}；"
                    f"路径长度={_path_length(failed_path)}；"
                    f"系统错误码={getattr(error, 'winerror', None) or error.errno}；原因={error}"
                )
            current["relax_mc_batches"][bid]["return_error"] = (
                f"回收未完成；回传包={result_archive(record)}；"
                f"请保留回传包，不要整批重投计算；详情：{detail}"
            )
            current["relax_mc_batches"][bid]["return_diagnostic"] = {
                "origin": "local_import",
                "stage": "archive_validation_or_import",
                "code": type(error).__name__,
                "error": detail[:4096],
                "evidence_path": str(result_archive(record)),
            }
    root = (context.get("effective_config") or {}).get("upload_batches_directory")
    if root and current.get("relax_mc_batches"):
        from phase_agent.tools.remote.submission_layout import write_submission_index

        write_submission_index(root, current)
    return current

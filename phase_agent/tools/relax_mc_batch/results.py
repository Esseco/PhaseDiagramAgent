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
    if model != record["model_version"] or context["config_version"] != record["config_version"]:
        raise ValueError(
            "batch_model_or_config_version_mismatch; inspect historical results without activating them"
        )
    archive = result_archive(record)
    digest = checksum(archive)
    if record.get("import_checksum") == digest:
        return {
            "status": "already_imported",
            "state": state if _owned else deepcopy(state),
            "submitted": False,
        }
    # A changed return is transactional even for an owned caller: failed
    # validation must not leave partial task/budget updates in its input.
    state = deepcopy(state)
    record = state["relax_mc_batches"][bundle_id]
    # Preserve extended paths through receipt checks and final-geometry reads,
    # not only ZIP staging: HPC member names can exceed Windows' legacy limit.
    destination = local_path(Path(config["upload_batches_directory"]) / "verified" / digest[:16])
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
    )
    state["relax_mc_batches"][bundle_id].pop("return_error", None)
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


def recover_available(state, context, *, _owned=False):
    current = state if _owned else deepcopy(state)
    for bid, record in list((current.get("relax_mc_batches") or {}).items()):
        if record.get("status") in {"cancelled", "superseded"}:
            continue
        if not result_archive(record).is_file():
            status_file = (
                output_directory(record["directory"]) / "run_status.json"
                if record.get("directory")
                else None
            )
            diagnostic = status_file.with_name("diagnostics.zip") if status_file else None
            if status_file is not None and (status_file.is_file() or diagnostic.is_file()):
                try:
                    if status_file.is_file():
                        status = read(status_file)
                    else:
                        with zipfile.ZipFile(diagnostic) as archive:
                            if archive.getinfo("run_status.json").file_size > 65536:
                                raise ValueError("oversized diagnostic status")
                            status = json.loads(archive.read("run_status.json"))
                    record["return_error"] = (
                        "回传文件夹缺少 results.zip；计算节点退出码 "
                        + str(status.get("returncode"))
                        + "；"
                        + str(status.get("error") or "请查看 diagnostics.zip 内 run.log")
                    )
                except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
                    record["return_error"] = "回传状态文件无效：" + str(error)
            continue
        try:
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
    return current

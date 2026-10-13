"""Verify, inspect and merge batch returns through existing ledger/reconciliation APIs."""

from copy import deepcopy
from pathlib import Path
import tempfile
import zipfile

from phase_agent.tools.remote.transfer_bundle import import_bundle, checksum
from phase_agent.tools.relax_mc_batch.files import read, rebase, PATH_KEYS
from phase_agent.tools.state.reconcile_task_results import reconcile_task_results
from phase_agent.tools.workflows.apply_scientific_feedback import apply_scientific_feedback
from phase_agent.tools.budget.reserve_budget import reserve_budget


def inspect_archive(path):
    # Full checksum/path verification uses the same existing transfer implementation.
    with tempfile.TemporaryDirectory(prefix="r-") as temporary:
        import_bundle(path, temporary, extra_result_suffixes=(".csv", ".bin"))
        return {
            "summary": read(Path(temporary) / "result_summary.json"),
            "candidate_pool": read(Path(temporary) / "candidate_pool.json"),
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


def import_results(*, action, context):
    state = deepcopy(context.get("event_state") or {})
    bundle_id = (action.get("parameters") or {}).get("bundle_id")
    record = (state.get("relax_mc_batches") or {}).get(bundle_id)
    if not record:
        raise ValueError("unknown_registered_relax_mc_bundle")
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
    archive = Path(record["expected_results_archive"])
    digest = checksum(archive)
    if record.get("import_checksum") == digest:
        return {"status": "already_imported", "state": state, "submitted": False}
    # Keep nested scientific artifact paths below Windows' legacy path limit.
    destination = Path(config["upload_batches_directory"]) / "verified" / digest[:16]
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
            if task.get("stage") != "deep_search" or task.get("segment_index") not in (0, 1, 2):
                raise ValueError("unexpected_remote_task")
            state.setdefault("tasks", []).append({**deepcopy(task), "status": "pending"})
            reserved = reserve_budget(
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
    )
    state["relax_mc_batches"][bundle_id].pop("return_error", None)
    return {
        "status": "completed",
        "state": state,
        "result_summary": summary,
        "candidate_pool": candidate,
        "submitted": False,
    }


def recover_available(state, context):
    current = deepcopy(state)
    for bid, record in list((current.get("relax_mc_batches") or {}).items()):
        if not Path(record.get("expected_results_archive") or "").is_file():
            continue
        try:
            result = import_results(
                action={"parameters": {"bundle_id": bid}},
                context={**context, "event_state": current},
            )
            current = result["state"]
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
            current["relax_mc_batches"][bid]["return_error"] = str(error)
    return current

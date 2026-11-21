"""Identify same-frame MC energy corrections with auditable source bindings."""

from math import isfinite
from pathlib import Path

from phase_agent.tools.remote.integrity import file_checksum


def verified_mc_energy_repair(task):
    output = task.get("outputs") or {}
    params = task.get("parameters") or {}
    geometry = task.get("input_structure_sha256")
    model = task.get("model_sha256")
    energy = output.get("energy")
    return bool(
        task.get("stage") == "relax_and_feature"
        and task.get("status") == "completed"
        and task.get("checks_passed", True) is True
        and task.get("energy_basis_repair_id")
        and task.get("energy_repair_source_task_id")
        and task.get("energy_repair_source_record_id")
        and task.get("parent_decision_id")
        and params.get("energy_basis_repair") is True
        and params.get("model_refresh_operation") == "predict"
        and output.get("single_point_completed") is True
        and output.get("energy_source") == "main_model_selected_frame"
        and output.get("geometry") == "mc_energy_repair_same_frame"
        and isinstance(geometry, str)
        and len(geometry) == 64
        and output.get("structure_checksum") == geometry
        and isinstance(model, str)
        and len(model) == 64
        and output.get("verified_model_sha256") == model
        and isinstance(energy, (int, float))
        and not isinstance(energy, bool)
        and isfinite(energy)
        and output.get("energy_unit") == "eV"
    )



def cross_model_repair_binding_valid(state, task):
    """Verify a v2 same-frame repair against its frozen source and target."""
    from phase_agent.tools.remote.integrity import payload_checksum

    repair_id = task.get("energy_basis_repair_id")
    repair = (state.get("mc_energy_repairs") or {}).get(repair_id) or {}
    plan = repair.get("preview") or {}
    if plan.get("schema") != 2 or plan.get("checksum") != repair_id:
        return False
    if payload_checksum({k: v for k, v in plan.items() if k != "checksum"}) != repair_id:
        return False
    frames = [r for r in plan.get("frames") or []
              if r.get("source_task_id") == task.get("energy_repair_source_task_id")]
    if len(frames) != 1:
        return False
    frame = frames[0]
    source = next((t for t in state.get("tasks") or []
                   if t.get("task_id") == frame.get("source_task_id")), {})
    old = next((r for r in state.get("phase_records") or []
                if r.get("record_id") == frame.get("record_id")), {})
    model = plan.get("model") or {}
    version = frame.get("source_model_version")
    path = Path((task.get("outputs") or {}).get("structure_path") or "")
    valid = (
        version and task.get("task_id") in (repair.get("task_ids") or [])
        and task.get("parent_decision_id") == repair.get("approval_record_id")
        and source.get("stage") == "deep_search" and source.get("status") == "completed"
        and source.get("model_version") == old.get("model_version") == version
        and task.get("energy_repair_source_model_version") == version
        and old.get("source_task_id") == frame.get("source_task_id")
        and task.get("energy_repair_source_record_id") == frame.get("record_id")
        and source.get("structure_id") == old.get("structure_id") == frame.get("structure_id") == task.get("structure_id")
        and (source.get("outputs") or {}).get("structure_checksum")
            == old.get("structure_sha256") == frame.get("structure_sha256") == task.get("input_structure_sha256")
        and task.get("model_version") == model.get("version")
        and task.get("model_sha256") == model.get("sha256")
        and verified_mc_energy_repair(task) and path.is_file()
    )
    if not valid:
        return False
    try:
        return file_checksum(path) == frame.get("structure_sha256")
    except OSError:
        return False


def missing_verified_repair_tasks(state):
    """Find already-processed corrections whose phase feedback was missed."""
    processed = set(state.get("feedback_processed_task_ids") or [])
    existing = {r.get("source_task_id") for r in state.get("phase_records") or []}
    records = {r.get("record_id"): r for r in state.get("phase_records") or []}
    tasks = {t.get("task_id"): t for t in state.get("tasks") or []}
    result = []
    for repair_id, repair in (state.get("mc_energy_repairs") or {}).items():
        preview = repair.get("preview") or {}
        model = preview.get("model") or {}
        if preview.get("checksum") != repair_id:
            continue
        frames = {f.get("source_task_id"): f for f in preview.get("frames") or []}
        for task_id in repair.get("task_ids") or []:
            if task_id not in processed or task_id in existing:
                continue
            task = tasks.get(task_id) or {}
            frame = frames.get(task.get("energy_repair_source_task_id")) or {}
            prior = records.get(task.get("energy_repair_source_record_id")) or {}
            if preview.get("schema") == 2:
                if not cross_model_repair_binding_valid(state, task):
                    continue
                source = tasks.get(task.get("energy_repair_source_task_id")) or {}
                source_version = frame.get("source_model_version")
                if not (source_version and source.get("stage") == "deep_search"
                        and source.get("model_version") == source_version
                        and prior.get("model_version") == source_version
                        and task.get("energy_repair_source_model_version") == source_version
                        and (source.get("outputs") or {}).get("structure_checksum")
                        == frame.get("structure_sha256")):
                    continue
            out = task.get("outputs") or {}
            path = Path(out.get("structure_path") or "")
            if not (
                verified_mc_energy_repair(task)
                and task.get("energy_basis_repair_id") == repair_id
                and task.get("parent_decision_id") == repair.get("approval_record_id")
                and frame.get("record_id") == task.get("energy_repair_source_record_id")
                and frame.get("structure_id") == task.get("structure_id")
                and frame.get("structure_sha256") == task.get("input_structure_sha256")
                and prior.get("source_task_id") == task.get("energy_repair_source_task_id")
                and prior.get("structure_sha256") == task.get("input_structure_sha256")
                and task.get("model_version") == model.get("version")
                and task.get("model_sha256") == model.get("sha256")
                and path.is_file()
            ):
                continue
            try:
                if file_checksum(path) != task["input_structure_sha256"]:
                    continue
            except OSError:
                continue
            result.append(task)
    return result


def superseded_mc_energy_record_ids(records, tasks):
    """Only verified same-frame corrections replace their exact legacy source."""
    by_id = {r.get("record_id"): r for r in records}
    by_task = {t.get("task_id"): t for t in tasks}
    superseded = set()
    for record in records:
        task = by_task.get(record.get("source_task_id")) or {}
        if not verified_mc_energy_repair(task):
            continue
        old = by_id.get(task.get("energy_repair_source_record_id")) or {}
        if (
            record.get("status", "completed") == "completed"
            and record.get("energy_method") == "mlip"
            and record.get("structure_id") == old.get("structure_id") == task.get("structure_id")
            and record.get("structure_sha256")
            == old.get("structure_sha256")
            == task.get("input_structure_sha256")
            and record.get("model_version") == old.get("model_version") == task.get("model_version")
            and old.get("source_task_id") == task.get("energy_repair_source_task_id")
        ):
            superseded.add(old["record_id"])
    return superseded


def stale_mc_energy_model_snapshots(state):
    """Return model snapshots that still contain already corrected source records."""
    superseded = superseded_mc_energy_record_ids(
        state.get("phase_records") or [], state.get("tasks") or []
    )
    return [
        version
        for version, snapshot in (state.get("phase_diagrams_by_model") or {}).items()
        if any(row.get("record_id") in superseded for row in snapshot.get("entries") or [])
    ]

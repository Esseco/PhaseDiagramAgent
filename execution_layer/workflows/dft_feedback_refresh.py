"""Synchronize a reassessed DFT task into existing scientific records, not costs."""
from copy import deepcopy

SCIENCE_KEYS = ("magnetic_moments", "magnetic_check", "spin_state_check", "actual_phase", "phase_identification",
                "remote_mlip_prediction", "mlip_result_file", "mlip_result_checksum")


def sync_dft_assessment(state, result, manager):
    outputs = result.get("outputs") or {}
    task_id = result.get("task_id")
    for task in state.get("tasks", []):
        if task.get("task_id") == task_id:
            for key in ("checks_passed", "checks_passed_before_spin", "quality_rejection_reasons"):
                if key in result:
                    task[key] = deepcopy(result[key])
            task.setdefault("outputs", {}).update({key: deepcopy(outputs[key]) for key in SCIENCE_KEYS if key in outputs})
    for record in state.get("phase_records", []):
        if record.get("source_task_id") == task_id:
            record["checks_passed"] = result.get("checks_passed", True)
            record.update({key: deepcopy(outputs[key]) for key in SCIENCE_KEYS if key in outputs})
    structure = manager.data.get("structures", {}).get(result.get("structure_id")) or {}
    for row in (structure.get("stage_history") or {}).get(result.get("stage"), []):
        metadata = row.get("metadata") or {}
        if metadata.get("task_id") == task_id:
            metadata.update({"checks_passed": result.get("checks_passed", True),
                             **{key: deepcopy(outputs[key]) for key in SCIENCE_KEYS if key in outputs}})
            row["metadata"] = metadata


def assessment_changed(state, result):
    saved = next((row for row in state.get("dft_dataset_records", []) if row.get("task_id") == result.get("task_id")), None)
    if saved is None:
        return True
    outputs = result.get("outputs") or {}
    return (saved.get("checks_passed", True) != result.get("checks_passed", True)
            or any(saved.get(key) != outputs.get(key) for key in SCIENCE_KEYS))

"""回收计算结果并按真实状态写入现有台账。"""

from __future__ import annotations

import copy
from typing import Any


def collect_calculation_results(
    manager: Any,
    structure_id: str,
    result: dict[str, Any],
    *,
    ledger_path: str | None = None,
    refresh_existing: bool = False,
) -> dict[str, Any]:
    """完成项写 stage_history；其他状态只写 calculation_attempts。"""
    structure = manager.data["structures"].get(structure_id)
    if structure is None:
        raise KeyError(f"未知结构：{structure_id}")
    stage = result.get("stage")
    stages = getattr(manager, "stages", manager.STAGES)
    if stage not in stages:
        raise ValueError(f"未知阶段：{stage}")
    status = result.get("status")
    allowed = {
        "pending",
        "running",
        "completed",
        "failed",
        "timeout",
        "cancelled",
        "paused",
        "not_configured",
    }
    if status not in allowed:
        raise ValueError(f"未知任务状态：{status}")
    metadata = structure.get("metadata")
    if metadata is None:
        metadata = {}
        structure["metadata"] = metadata
    attempts = metadata.setdefault("calculation_attempts", [])
    attempt = {
        "task_id": result.get("task_id"),
        "stage": stage,
        "status": status,
        "converged": result.get("converged"),
        "actual_cost": copy.deepcopy(result.get("actual_cost")),
        "error": copy.deepcopy(result.get("error")),
        "checkpoint": result.get("checkpoint"),
        "result_path": result.get("result_path"),
        "model_version": result.get("model_version", result.get("calculation_version")),
        "parameters": copy.deepcopy(result.get("parameters") or {}),
        "checks_passed": result.get("checks_passed", True),
        "spin_state_check": copy.deepcopy((result.get("outputs") or {}).get("spin_state_check")),
        "magnetic_check": copy.deepcopy((result.get("outputs") or {}).get("magnetic_check")),
        "magnetic_moments": copy.deepcopy((result.get("outputs") or {}).get("magnetic_moments")),
        "budget": copy.deepcopy(result.get("budget") or {}),
    }
    same_task = next(
        (
            index
            for index, item in enumerate(attempts)
            if attempt["task_id"] is not None and item.get("task_id") == attempt["task_id"]
        ),
        None,
    )
    if same_task is None:
        if attempt not in attempts:
            attempts.append(attempt)
    else:
        attempts[same_task] = attempt
    result_id = None
    phase_record = None
    if status == "completed" and result.get("checks_passed", True) is True:
        outputs = result.get("outputs") or {}
        existing = (
            next(
                (
                    row
                    for row in (structure.get("stage_history") or {}).get(stage, [])
                    if (row.get("metadata") or {}).get("task_id") == result.get("task_id")
                ),
                None,
            )
            if refresh_existing
            else None
        )
        if existing:
            result_id = existing["result_id"]
        else:
            result_id = manager.record_result(
                structure_id=structure_id,
                stage=stage,
                converged=result.get("converged"),
                convergence_info=outputs.get("convergence_info"),
                mlip_name=outputs.get("mlip_name"),
                mlip_version=outputs.get("mlip_version"),
                mlip_relax_version=outputs.get("mlip_relax_version"),
                mlip_energy=outputs.get("energy")
                if stage in {"relax_and_feature", "deep_search"}
                else None,
                dft_code=outputs.get("dft_code"),
                dft_version=outputs.get("dft_version"),
                dft_settings=outputs.get("dft_settings"),
                dft_energy=outputs.get("energy")
                if stage in {"dft_single_point", "dft_relax"}
                else None,
                energy_unit=outputs.get("energy_unit"),
                ehull=outputs.get("ehull"),
                ehull_unit=outputs.get("ehull_unit"),
                hull_reference_version=outputs.get("hull_reference_version"),
                result_path=result.get("result_path"),
                metadata={
                    "task_id": result.get("task_id"),
                    "budget": copy.deepcopy(result.get("budget") or {}),
                    "model_version": result.get("model_version", result.get("calculation_version")),
                    "actual_cost": result.get("actual_cost"),
                    "features": outputs.get("features"),
                    "branch_score": outputs.get("branch_score"),
                    "source_phase": outputs.get("source_phase"),
                    "actual_phase": outputs.get("actual_phase"),
                    "phase_identification": outputs.get("phase_identification"),
                    "checks_passed": result.get("checks_passed", True),
                    "spin_state_check": copy.deepcopy(outputs.get("spin_state_check")),
                    "magnetic_check": copy.deepcopy(outputs.get("magnetic_check")),
                },
            )
        phase_record = _phase_record(manager, structure_id, result_id, stage, result)
    if ledger_path is not None:
        manager.save(ledger_path)
    return {
        "structure_id": structure_id,
        "stage": stage,
        "status": status,
        "result_id": result_id,
        "phase_record": phase_record,
    }


def _phase_record(manager, structure_id, result_id, stage, result):
    if result.get("checks_passed", True) is not True:
        return None
    outputs = result.get("outputs") or {}
    energy = outputs.get("energy")
    evaluated_refresh = (
        stage == "relax_and_feature"
        and result.get("model_refresh_id")
        and (result.get("parameters") or {}).get("model_refresh_operation") == "predict"
        and outputs.get("single_point_completed") is True
    )
    if energy is None or (
        stage != "deep_search" and result.get("converged") is not True and not evaluated_refresh
    ):
        return None
    unit = outputs.get("energy_unit")
    if unit != "eV":
        raise ValueError(f"相图能量必须明确使用 eV，当前为 {unit!r}")
    method = (
        "dft"
        if stage in {"dft_single_point", "dft_relax"}
        else "mlip"
        if stage in {"relax_and_feature", "deep_search"}
        else None
    )
    if method is None:
        return None
    structure = manager.data["structures"][structure_id]
    branch = manager.data["branches"][structure["branch_id"]]
    composition = (
        outputs.get("composition") or structure.get("composition") or branch.get("composition")
    )
    if not composition:
        return None
    source_version = (
        (
            result.get("model_version")
            or outputs.get("mlip_version")
            or outputs.get("mlip_relax_version")
        )
        if method == "mlip"
        else (outputs.get("dft_version") or "atomate")
    )
    identification = outputs.get("phase_identification") or {}
    identified = identification.get("status") == "identified" and bool(outputs.get("actual_phase"))
    return {
        "record_id": result_id,
        "structure_id": structure_id,
        "structure_path": outputs.get("structure_path") or outputs.get("final_structure_path"),
        "structure": copy.deepcopy(outputs.get("structure")),
        "final_frame_valid": outputs.get("final_frame_valid"),
        "phase": outputs.get("actual_phase") if identified else None,
        "source_phase": branch.get("P"),
        "phase_identification_status": "identified" if identified else "unknown",
        "phase_identification": copy.deepcopy(identification),
        "structure_sha256": identification.get("structure_sha256"),
        "composition": composition,
        "energy": float(energy),
        "energy_unit": unit,
        "energy_method": method,
        "checks_passed": result.get("checks_passed", True),
        "spin_state_check": copy.deepcopy(outputs.get("spin_state_check")),
        "magnetic_moments": copy.deepcopy(outputs.get("magnetic_moments")),
        "source_version": source_version,
        "model_version": source_version if method == "mlip" else None,
        "source_task_id": result.get("task_id"),
        "stage": stage,
        "status": "completed" if identified else "pending_phase_identification",
    }

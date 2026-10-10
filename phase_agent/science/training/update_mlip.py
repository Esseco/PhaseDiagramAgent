"""整理合格 DFT 数据并通过适配入口微调 MLIP。"""

from __future__ import annotations

from typing import Any, Callable


def update_mlip(
    dft_records: list[dict[str, Any]],
    *,
    historical_data: list[dict[str, Any]] | None = None,
    trainer: Callable[..., dict[str, Any]] | None = None,
    base_model: dict[str, Any] | None = None,
    training_config: dict[str, Any] | None = None,
    dataset_version: str,
    candidate_model_version: str,
) -> dict[str, Any]:
    """仅使用完成、收敛且含真实 DFT 能量的数据；不自动激活模型。"""
    accepted, rejected = [], []
    from phase_agent.science.dft.spin_acceptance import spin_standard_passed

    for record in dft_records:
        valid = (
            record.get("status") == "completed"
            and record.get("converged") is True
            and record.get("energy") is not None
            and record.get("checks_passed", True) is True
            and spin_standard_passed(record)
        )
        (accepted if valid else rejected).append(record)
    historical = [
        row
        for row in historical_data or []
        if row.get("checks_passed", True) is True and spin_standard_passed(row)
    ]
    dataset = _unique([*historical, *accepted])
    metadata = {
        "dataset_version": dataset_version,
        "candidate_model_version": candidate_model_version,
        "base_model_version": (base_model or {}).get("version"),
        "new_record_count": len(accepted),
        "total_record_count": len(dataset),
    }
    minimum = int(
        (training_config or {})
        .get("training", {})
        .get("minimum_new_dft_records", (training_config or {}).get("minimum_new_dft_records", 1))
    )
    if len(accepted) < minimum:
        return {
            "status": "insufficient_new_data",
            "candidate_model": None,
            "dataset": dataset,
            "metadata": {**metadata, "minimum_new_dft_records": minimum},
            "rejected": rejected,
            "error": None,
        }
    if trainer is None:
        return {
            "status": "not_configured",
            "candidate_model": None,
            "dataset": dataset,
            "metadata": metadata,
            "rejected": rejected,
            "error": "MLIP training adapter 未配置",
        }
    try:
        trained = trainer(
            dataset=dataset,
            base_model=base_model,
            config=dict(training_config or {}),
            metadata=metadata,
        )
        return {
            "status": "trained_candidate",
            "candidate_model": trained,
            "dataset": dataset,
            "metadata": metadata,
            "rejected": rejected,
            "activated": False,
            "error": None,
        }
    except Exception as error:
        return {
            "status": "failed",
            "candidate_model": None,
            "dataset": dataset,
            "metadata": metadata,
            "rejected": rejected,
            "activated": False,
            "error": f"{type(error).__name__}: {error}",
        }


def _unique(records):
    output, seen = [], set()
    for index, record in enumerate(records):
        key = record.get("data_id", record.get("structure_id", f"row:{index}"))
        if key not in seen:
            seen.add(key)
            output.append(record)
    return output

"""Extract measured GPU-hours only from explicit scheduler accounting fields."""

from __future__ import annotations


def extract_gpu_accounting(result: dict) -> dict:
    direct = result.get("actual_gpu_core_hours")
    if _number(direct):
        return {"actual_gpu_core_hours": max(0.0, float(direct)),
                "source": result.get("gpu_cost_source") or "reported_result",
                "measured": True}
    record = result.get("job_accounting") or result.get("scheduler_accounting") or {}
    if not isinstance(record, dict):
        return {"actual_gpu_core_hours": None, "source": None, "measured": False}
    source = str(record.get("source") or "").lower()
    reliable = record.get("reliable") is True or source in {
        "sacct", "slurm_sacct", "scheduler_accounting", "verified_scheduler_record"
    }
    gpu_count = record.get("gpu_count", record.get("allocated_gpus"))
    seconds = record.get("elapsed_seconds", record.get("runtime_seconds"))
    if reliable and _number(gpu_count) and _number(seconds):
        value = max(0.0, float(gpu_count)) * max(0.0, float(seconds)) / 3600.0
        return {"actual_gpu_core_hours": value, "source": source or "scheduler_accounting",
                "measured": True, "gpu_count": float(gpu_count),
                "elapsed_seconds": float(seconds), "job_id": record.get("job_id")}
    return {"actual_gpu_core_hours": None, "source": source or None, "measured": False}


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)

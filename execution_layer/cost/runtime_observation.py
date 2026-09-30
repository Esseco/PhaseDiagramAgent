"""One start/end observation; no polling, scheduler calls or scientific changes."""
import os
import platform
import time


def start_timer():
    return time.monotonic(), time.time()


def runtime_observation(start, task, result):
    monotonic_start, epoch_start = start
    env = os.environ
    def count(name):
        try:
            value = int(env.get(name, ""))
            return value if value > 0 else None
        except ValueError:
            return None
    params = (task.get("worker_job") or {}).get("parameters") or task.get("parameters") or {}
    return {"source": "task_start_end", "elapsed_seconds": max(0.0, time.monotonic() - monotonic_start),
            "started_at_epoch": epoch_start, "ended_at_epoch": time.time(),
            "backend": (task.get("atomate") or {}).get("backend") or (task.get("worker_job") or {}).get("backend") or task.get("model_version"),
            "hardware": env.get("SLURM_JOB_PARTITION") or platform.machine(),
            "gpu_count": count("SLURM_GPUS_ON_NODE"), "cpu_count": count("SLURM_CPUS_ON_NODE"),
            "job_id": env.get("SLURM_JOB_ID"), "batch_id": task.get("batch_id"),
            "allocation_basis": "task_time_slice_not_whole_batch",
            "patience": result.get("patience_steps", task.get("patience_steps", params.get("patience_steps", params.get("patience")))),
            "max_mc_steps": result.get("requested_max_mc_steps") or result.get("max_mc_steps") or task.get("max_mc_steps") or params.get("max_mc_steps") or params.get("max_steps"),
            "actual_mc_steps": result.get("actual_mc_steps"),
            "note": "Partition is an approximate hardware class, not verified GPU model. No queue time included."}

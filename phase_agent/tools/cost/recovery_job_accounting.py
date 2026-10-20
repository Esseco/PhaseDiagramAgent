"""Account verified recovery job clocks once, never assign whole-job hours to tasks."""

import math


def add_recovered_job_records(records, state):
    for bundle, recovery in (state.get("result_recoveries") or {}).items():
        job = recovery.get("job_evidence") or {}
        seconds, count = job.get("elapsed_seconds"), job.get("allocated_gpus")
        if (
            recovery.get("status") != "completed"
            or job.get("source") != "sacct"
            or not job.get("job_id")
        ):
            continue
        if any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
            for v in (seconds, count)
        ):
            continue
        job_id = str(job["job_id"])
        cluster = job.get("cluster") or next(
            (r.get("cluster", "") for r in records.values() if str(r.get("job_id")) == job_id), ""
        )
        key = f"job:{cluster}:{job_id}:0"
        records[key] = {
            "kind": "job",
            "job_id": job_id,
            "cluster": cluster,
            "stage": "deep_search",
            "elapsed_seconds": seconds,
            "gpu_count": count,
            "gpu_hours": seconds * count / 3600,
            "source": "sacct",
            "round_id": bundle,
        }
        for identity, record in records.items():
            if any(
                identity.startswith("task:" + task_id + ":")
                for task_id in recovery.get("recovered_task_ids") or []
            ):
                record["job_id"] = job_id
                record["cluster"] = cluster

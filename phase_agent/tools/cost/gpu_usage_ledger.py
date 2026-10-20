"""Idempotent GPU-hours ledger; job clocks override their task time slices."""

import csv, json, math, os
from pathlib import Path


def valid(value):
    return (
        isinstance(value, (float, int))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def round_id(path):
    if not path:
        return "unknown_round"
    parts = Path(path).resolve().parts
    for marker in ("results", "inputs"):
        if marker in parts:
            end = parts.index(marker)
            start = parts.index("submissions") + 1 if "submissions" in parts else max(0, end - 1)
            return "/".join(parts[start:end]) or "unknown_round"
    return Path(path).parent.name or "unknown_round"


def round_rows(records):
    import re

    rounds = {}
    whole = {
        (r.get("cluster", ""), str(r["job_id"]))
        for r in records.values()
        if r["kind"] == "job" and valid(r.get("gpu_hours"))
    }
    for row in records.values():
        key = row.get("round_id", "unknown_round")
        group = rounds.setdefault(
            key,
            {
                "round_id": key,
                "stages": set(),
                "record_count": 0,
                "known_gpu_hours": 0.0,
                "missing_time_records": 0,
            },
        )
        started = row.get("started_at_epoch")
        if valid(started):
            group["first_started_at"] = min(group.get("first_started_at", started), started)
        group["stages"].add(row["stage"])
        group["record_count"] += 1
        if row.get("included_in_total"):
            group["known_gpu_hours"] += row["gpu_hours"]
        covered = (row.get("cluster", ""), str(row.get("job_id"))) in whole or (
            not row.get("cluster") and any(job == str(row.get("job_id")) for _, job in whole)
        )
        if row.get("gpu_hours") is None and not (row["kind"] == "task_slice" and covered):
            group["missing_time_records"] += 1
    cumulative = 0.0
    natural = lambda value: [
        int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)
    ]
    priorities = {
        "relax_and_feature": 0,
        "deep_search": 1,
        "dft_relax": 2,
        "dft_single_point": 3,
        "mlip_training": 4,
    }

    def order(value):
        epoch = re.search(r"epoch(\d+)", value)
        group = rounds[value]
        return (
            int(epoch.group(1)) if epoch else float("inf"),
            group.get("first_started_at", -1),
            min(priorities.get(s, 9) for s in group["stages"]),
            natural(value),
        )

    for key in sorted(rounds, key=order):
        group = rounds[key]
        cumulative += group["known_gpu_hours"]
        if group["known_gpu_hours"] == 0 and group["missing_time_records"] == group["record_count"]:
            group["known_gpu_hours"] = None
        yield {
            **group,
            "stages": ",".join(sorted(group["stages"])),
            "cumulative_known_gpu_hours": cumulative,
        }


def summarize(records):
    whole = {
        (r.get("cluster", ""), str(r["job_id"]))
        for r in records.values()
        if r["kind"] == "job" and valid(r.get("gpu_hours"))
    }
    scheduler = {
        (r.get("cluster", ""), str(r["job_id"]))
        for r in records.values()
        if r["kind"] == "job" and r.get("source") == "sacct" and valid(r.get("gpu_hours"))
    }
    included = []
    unknown = 0
    for key, row in records.items():
        covered = (row.get("cluster", ""), str(row.get("job_id"))) in whole or (
            not row.get("cluster") and any(job == str(row.get("job_id")) for _, job in whole)
        )
        row["included_in_total"] = valid(row.get("gpu_hours")) and (
            row["kind"] == "job" or not covered
        )
        if (
            row["kind"] == "job"
            and row.get("source") != "sacct"
            and (row.get("cluster", ""), str(row.get("job_id"))) in scheduler
        ):
            row["included_in_total"] = False
        if row["included_in_total"]:
            included.append(row)
        if row.get("gpu_hours") is None and not (row["kind"] == "task_slice" and covered):
            unknown += 1
    stages = {}
    for row in included:
        stages[row["stage"]] = stages.get(row["stage"], 0) + row["gpu_hours"]
    return {
        "total_gpu_hours": sum(r["gpu_hours"] for r in included),
        "by_stage": stages,
        "record_count": len(records),
        "accounted_records": len(included),
        "unknown_records": unknown,
        "unit": "GPU hour",
        "basis": "allocated GPU count multiplied by elapsed time; excludes queue time; whole jobs override task slices",
    }


def update_gpu_usage(state, config, results=()):
    state_path = config.get("state_path")
    if not state_path:
        return None
    ledger = Path(state_path).parent / "ledgers/gpu_usage.json"
    data = (
        json.loads(ledger.read_text(encoding="utf-8"))
        if ledger.exists()
        else {"schema": 1, "records": {}}
    )
    records = data["records"]
    from phase_agent.tools.remote.transfer_bundle import registered_result_directories

    roots = {Path(p) for p in registered_result_directories(state)}
    for job in (state.get("remote_finetune_jobs") or {}).values():
        if job.get("directory"):
            roots.add(Path(job["directory"]) / "results")
    errors = []
    for root in roots:
        for path in root.glob("gpu-job-*.runtime.json"):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
                if row.get("allocation_basis") != "whole_gpu_job" or not row.get("job_id"):
                    continue
                seconds = row.get("elapsed_seconds")
                count = row.get("gpu_count")
                hours = seconds * count / 3600 if valid(seconds) and valid(count) else None
                key = f"job:{row.get('cluster', '')}:{row['job_id']}:{row.get('restart_count', 0)}"
                incoming = {
                    "kind": "job",
                    "job_id": str(row["job_id"]),
                    "cluster": row.get("cluster", ""),
                    "stage": row.get("stage", "unknown"),
                    "elapsed_seconds": seconds,
                    "gpu_count": count,
                    "gpu_hours": hours,
                    "source": row.get("source"),
                    "exit_code": row.get("exit_code"),
                    "restart_count": str(row.get("restart_count", 0)),
                }
                prior = records.get(key)
                incoming["round_id"] = round_id(root)
                incoming["started_at_epoch"] = row.get("started_at_epoch")
                if prior:
                    prior["round_id"] = incoming["round_id"]
                if prior and prior.get("source") == "sacct" and incoming["source"] != "sacct":
                    continue
                if (
                    prior
                    and valid(prior.get("gpu_hours"))
                    and (
                        hours is None
                        or (incoming["source"] != "sacct" and hours < prior["gpu_hours"])
                    )
                ):
                    continue
                records[key] = incoming
            except (OSError, ValueError, TypeError) as error:
                errors.append({"file": str(path), "error": str(error)})
    for task in [*state.get("tasks", []), *results]:
        if task.get("shared_dft_record_id"):
            continue
        observation = task.get("runtime_observation") or {}
        missing_key = f"task:{task.get('task_id')}:missing"
        if not observation:
            if task.get("stage") in {
                "relax_and_feature",
                "deep_search",
                "dft_relax",
                "dft_single_point",
            } and task.get("status") in {"completed", "failed", "timeout", "cancelled"}:
                records[missing_key] = {
                    "kind": "task_slice",
                    "job_id": str(task.get("slurm_job_id") or ""),
                    "cluster": "",
                    "stage": task["stage"],
                    "elapsed_seconds": None,
                    "gpu_count": None,
                    "gpu_hours": None,
                    "source": "missing_runtime",
                    "exit_code": None,
                    "restart_count": None,
                }
            if missing_key in records:
                records[missing_key]["round_id"] = round_id(
                    task.get("calculation_result_path")
                    or task.get("input_path")
                    or task.get("result_path")
                )
            continue
        records.pop(missing_key, None)
        if observation.get("allocation_basis") == "whole_batch_do_not_add_to_task_slices":
            continue
        seconds = observation.get("elapsed_seconds")
        count = observation.get("gpu_count")
        hours = seconds * count / 3600 if valid(seconds) and valid(count) else None
        key = f"task:{task.get('task_id')}:{observation.get('job_id')}:{observation.get('started_at_epoch')}"
        records[key] = {
            "kind": "task_slice",
            "job_id": str(observation.get("job_id", "")),
            "cluster": observation.get("cluster", ""),
            "stage": task.get("stage", "unknown"),
            "elapsed_seconds": seconds,
            "gpu_count": count,
            "gpu_hours": hours,
            "source": "task_start_end",
            "exit_code": None,
            "restart_count": None,
        }
        records[key]["started_at_epoch"] = observation.get("started_at_epoch")
    from phase_agent.tools.cost.recovery_job_accounting import add_recovered_job_records

    add_recovered_job_records(records, state)
    summary = summarize(records)
    summary["read_errors"] = errors
    labels = {
        str(task.get("task_id")): round_id(
            task.get("calculation_result_path") or task.get("input_path") or task.get("result_path")
        )
        for task in [*state.get("tasks", []), *results]
    }
    for key, row in records.items():
        if key.startswith("task:"):
            identity = key[5:].rsplit(":", 2)[0]
            if identity in labels:
                row["round_id"] = labels[identity]
    data["summary"] = summary
    state["gpu_usage_summary"] = summary
    ledger.parent.mkdir(parents=True, exist_ok=True)
    temp = ledger.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, ledger)
    output = Path(config.get("phase_diagram_directory") or ledger.parent) / "gpu_usage.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "round_id",
        "stages",
        "record_count",
        "known_gpu_hours",
        "missing_time_records",
        "cumulative_known_gpu_hours",
    ]
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(round_rows(records))
    return summary

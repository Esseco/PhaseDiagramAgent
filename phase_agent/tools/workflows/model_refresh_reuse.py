"""Exact, quality-checked reuse from an explicitly superseded refresh."""

import math
from collections import defaultdict


def archived_refresh(state, checksum):
    return next(
        (
            item["prior_refresh"]
            for item in reversed(state.get("completed_refresh_replans") or [])
            if ((item.get("prior_refresh") or {}).get("plan") or {}).get("checksum") == checksum
        ),
        None,
    )


def matching_refresh_results(state, plan, prior):
    model = plan.get("model") or {}
    old_model = (prior.get("plan") or {}).get("model") or {}
    if (
        not model.get("sha256")
        or model["sha256"] != old_model.get("sha256")
        or plan.get("new_model_version") != prior.get("new_model_version")
        or (plan.get("policy") or {}).get("force_followup_version") != 2
    ):
        return {}
    ids = set(prior.get("task_ids") or [])
    records = defaultdict(list)
    for record in state.get("phase_records") or []:
        if record.get("source_task_id") in ids:
            records[record["source_task_id"]].append(record)
    tasks = defaultdict(list)
    for task in state.get("tasks") or []:
        if task.get("task_id") in ids:
            key = (
                task.get("structure_id"),
                task.get("input_structure_sha256"),
                (task.get("parameters") or {}).get("model_refresh_operation"),
            )
            tasks[key].append(task)
    reuse = {}
    version = plan["new_model_version"]
    for row in plan["candidates"]:
        sha, op = row["structure_sha256"], row["operation"]
        matches = tasks[row["structure_id"], sha, op]
        if len(matches) != 1:
            continue
        task = matches[0]
        tid = task["task_id"]
        output = task.get("outputs") or {}
        params = task.get("parameters") or {}
        published = records[tid]
        if (
            task.get("status") != "completed"
            or task.get("checks_passed", True) is not True
            or task.get("result_disposition")
            or task.get("model_version") != version
            or output.get("mlip_version") != version
            or len(published) != 1
            or published[0].get("status", "completed") != "completed"
            or published[0].get("source_version") != version
            or published[0].get("structure_sha256") != output.get("structure_checksum")
            or not output.get("structure_checksum")
            or any(params.get(k) != v for k, v in plan.get("relax_parameters", {}).items())
        ):
            continue
        if op == "relax":
            valid = _relaxation_ok(task, output)
        else:
            valid = _prediction_ok(row, task, output, version)
        if valid:
            reuse[sha] = tid
    return reuse


def _relaxation_ok(task, output):
    return (
        task.get("converged") is True
        and output.get("optimizer_converged") is True
        and (output.get("extrapolation_monitor") or {}).get("extrapolation_failure") is False
    )


def _prediction_ok(row, task, output, version):
    evidence = output.get("refresh_force_followup") or {}
    initial = evidence.get("initial_prediction") or {}
    sha = row["structure_sha256"]
    threshold = 0.1 if row.get("ehull") is None or row["ehull"] <= 0.010 else 0.2
    force, vectors = evidence.get("force_max_ev_per_angstrom"), initial.get("forces")
    if (
        (task.get("parameters") or {}).get("refresh_force_threshold") != threshold
        or evidence.get("threshold_ev_per_angstrom") != threshold
        or evidence.get("input_structure_sha256") != sha
        or initial.get("structure_checksum") != sha
        or initial.get("single_point_completed") is not True
        or initial.get("mlip_version") != version
        or initial.get("forces_unit") != "eV/angstrom"
        or type(force) not in (int, float)
        or not math.isfinite(force)
        or not isinstance(vectors, list)
        or len(vectors) != row["atom_count"]
        or not vectors
        or not all(
            isinstance(v, list) and len(v) == 3
            and all(type(x) in (int, float) and math.isfinite(x) for x in v)
            for v in vectors
        )
        or not math.isclose(
            max(math.sqrt(sum(x * x for x in v)) for v in vectors),
            force, rel_tol=1e-12, abs_tol=1e-12,
        )
    ):
        return False
    if force > threshold:
        return evidence.get("relaxed") is True and _relaxation_ok(task, output)
    return (
        evidence.get("relaxed") is False
        and output.get("single_point_completed") is True
        and output.get("structure_checksum") == sha
    )

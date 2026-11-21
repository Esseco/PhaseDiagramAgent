"""Identify superseded input frames whose latest accepted relaxation leaves the search boundary."""


def outside_refresh_inputs(records, tasks, model_rounds):
    from phase_agent.analysis.convergence.mlip_relaxation import relaxation_failure

    counts = {}
    by_id = {}
    for task in tasks:
        tid = task.get("task_id")
        counts[tid] = counts.get(tid, 0) + 1
        by_id[tid] = task
    by_task = {}
    for record in records:
        tid = record.get("source_task_id")
        if tid:
            by_task.setdefault(tid, []).append(record)
    blocked = {}
    for tid, rows in by_task.items():
        task = by_id.get(tid) or {}
        if counts.get(tid) != 1 or len(rows) != 1:
            continue
        record = rows[0]
        output = task.get("outputs") or {}
        evidence = output.get("phase_identification") or {}
        version = task.get("model_version")
        epoch = model_rounds.get(version)
        sid = task.get("structure_id")
        input_hash = task.get("input_structure_sha256")
        monitor = output.get("extrapolation_monitor") or {}
        if (
            task.get("status") != "completed"
            or task.get("stage") != "relax_and_feature"
            or task.get("checks_passed", True) is not True
            or task.get("converged") is not True
            or relaxation_failure(task)
            or monitor.get("extrapolation_failure") is not False
            or (task.get("parameters") or {}).get("model_refresh_operation") not in {"predict", "relax"}
            or type(epoch) is not int or epoch < 1
            or not sid or not input_hash
            or record.get("checks_passed", True) is not True
            or record.get("energy_method") != "mlip"
            or record.get("status") != "outside_search_boundary"
            or record.get("structure_id") != sid
            or (record.get("model_version") or record.get("source_version")) != version
            or task.get("phase_identification_status") != "outside_boundary"
            or evidence.get("status") != "outside_boundary"
            or not evidence.get("observed_phase")
            or not evidence.get("classification_scope")
            or not evidence.get("structure_sha256")
            or evidence.get("structure_sha256") != record.get("structure_sha256")
            or evidence != record.get("phase_identification")
        ):
            continue
        key = (sid, input_hash)
        blocked[key] = max(epoch, blocked.get(key, 0))
    return blocked


def unidentified_refresh_inputs(records, tasks, model_rounds):
    """Do not revive an older input when its newer refresh has unknown phase."""
    from collections import Counter
    from phase_agent.analysis.phase.unidentified_mlip import verified_unidentified_mlip

    counts = Counter(t.get("task_id") for t in tasks)
    by_task = {t.get("task_id"): t for t in tasks}
    record_counts = Counter(r.get("source_task_id") for r in records)
    blocked = {}
    for record in records:
        tid = record.get("source_task_id")
        task = by_task.get(tid) or {}
        epoch = model_rounds.get(task.get("model_version"))
        sid = task.get("structure_id")
        input_hash = task.get("input_structure_sha256")
        if (counts.get(tid) != 1 or record_counts.get(tid) != 1
                or type(epoch) is not int or epoch < 1
                or not sid or not input_hash or record.get("structure_id") != sid
                or (task.get("parameters") or {}).get("model_refresh_operation") not in {"predict", "relax"}
                or not verified_unidentified_mlip(record, task)):
            continue
        key = (sid, input_hash)
        blocked[key] = max(epoch, blocked.get(key, 0))
    return blocked



def rejected_refresh_inputs(records, tasks, model_rounds, *, preserve_extrapolation=False):
    """Block an older input when its verified newer refresh is explicitly rejected."""
    from collections import Counter

    counts = Counter(t.get("task_id") for t in tasks)
    by_task = {}
    for record in records:
        by_task.setdefault(record.get("source_task_id"), []).append(record)
    blocked = {}
    for task in tasks:
        tid = task.get("task_id")
        matched = by_task.get(tid) or []
        output = task.get("outputs") or {}
        evidence = output.get("phase_identification") or {}
        epoch = model_rounds.get(task.get("model_version"))
        sid = task.get("structure_id")
        input_hash = task.get("input_structure_sha256")
        if (
            counts.get(tid) != 1
            or len(matched) > 1
            or task.get("status") != "completed"
            or task.get("stage") != "relax_and_feature"
            or task.get("checks_passed") is not False
            or (task.get("parameters") or {}).get("model_refresh_operation") not in {"predict", "relax"}
            or type(epoch) is not int or epoch < 1
            or not sid or not input_hash or not task.get("task_checksum")
        ):
            continue
        if matched:
            record = matched[0]
            if (record.get("structure_id") != sid
                or (record.get("model_version") or record.get("source_version")) != task.get("model_version")
                or record.get("checks_passed") is not False):
                continue
        extrapolation = (
            task.get("result_disposition") == "excluded_extrapolation_failure"
            and (output.get("extrapolation_monitor") or {}).get("extrapolation_failure") is True
            and (not matched or matched[0].get("status") == "excluded_extrapolation_failure")
        )
        unknown = (
            task.get("result_disposition") == "discarded_unknown_phase"
            and len(matched) == 1 and matched[0].get("status") == "excluded_unknown_phase"
            and evidence.get("status") == "unknown"
            and evidence.get("reason") == "ValueError: unrecognized_phase:X"
            and (task.get("phase_evidence_recovery") or {}).get("decision") == "discard_unknown"
        )
        if (not extrapolation or preserve_extrapolation) and not unknown:
            continue
        key = (sid, input_hash)
        blocked[key] = max(epoch, blocked.get(key, 0))
    return blocked

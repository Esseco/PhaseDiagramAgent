"""Backfill identified final phases into existing phase records without relabeling branches."""


def refresh_identified_phases(state):
    by_task = {}
    by_identity = {}
    for task in state.get("tasks") or []:
        if task.get("status") != "completed" or task.get("stage") not in {
            "relax_and_feature",
            "deep_search",
            "dft_single_point",
            "dft_relax",
        }:
            continue
        output = task.get("outputs") or {}
        if task.get("task_id"):
            by_task[task["task_id"]] = output
        if (
            output.get("actual_phase")
            and (output.get("phase_identification") or {}).get("status") == "identified"
            and output.get("energy") is not None
            and task.get("structure_id")
            and task.get("model_version")
        ):
            key = (task["structure_id"], task["model_version"], round(float(output["energy"]), 8))
            by_identity.setdefault(key, set()).add(output["actual_phase"])

    changed = False
    for record in state.get("phase_records") or []:
        if record.get("energy_method") not in {"mlip", "dft"} or record.get("energy") is None:
            continue
        key = (
            record.get("structure_id"),
            record.get("model_version") or record.get("source_version"),
            round(float(record["energy"]), 8),
        )
        matches = by_identity.get(key) or set()
        task_output = by_task.get(record.get("source_task_id")) or {}
        if (
            task_output.get("energy") is not None
            and abs(float(task_output["energy"]) - float(record["energy"])) < 1e-8
        ):
            actual = (
                task_output.get("actual_phase")
                if (task_output.get("phase_identification") or {}).get("status") == "identified"
                else None
            )
            identification = task_output.get("phase_identification") or {}
        else:
            actual = next(iter(matches)) if len(matches) == 1 else None
            identification = {}
            if actual is None and record.get("phase_identification_status") in {
                "identified",
                "outside_boundary",
            }:
                actual = (
                    record.get("phase")
                    if record.get("phase_identification_status") == "identified"
                    else None
                )
                identification = record.get("phase_identification") or {}
        outside = not actual and identification.get("status") == "outside_boundary"
        status = (
            "completed"
            if actual
            else "outside_search_boundary"
            if outside
            else "pending_phase_identification"
        )
        identification_status = (
            "identified" if actual else "outside_boundary" if outside else "unknown"
        )
        observed = actual or identification.get("observed_phase")
        digest = identification.get("structure_sha256") if identification else None
        if (
            record.get("phase") != actual
            or record.get("status") != status
            or record.get("phase_identification_status") != identification_status
            or record.get("observed_phase") != observed
            or (digest and record.get("structure_sha256") != digest)
        ):
            record.setdefault("source_phase", record.get("phase"))
            record["phase"] = actual
            record["actual_phase"] = actual
            record["observed_phase"] = observed
            record["status"] = status
            record["phase_identification_status"] = identification_status
            if identification:
                record["phase_identification"] = identification
                record["structure_sha256"] = digest
            changed = True
    return changed

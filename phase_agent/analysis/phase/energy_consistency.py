"""Identify saved hull points proven to use legacy MC committee-mean energies."""

import math


def legacy_mc_mean_record_ids(state, snapshot):
    records = {row.get("record_id"): row for row in state.get("phase_records") or []}
    tasks = {row.get("task_id"): row for row in state.get("tasks") or []}
    matched = []
    for entry in snapshot.get("entries") or []:
        record = records.get(entry.get("record_id")) or {}
        task = tasks.get(record.get("source_task_id")) or {}
        outputs = task.get("outputs") or {}
        qbc = outputs.get("qbc") or {}
        pool = outputs.get("pool_summary") or []
        if (
            task.get("stage") != "deep_search"
            or outputs.get("energy_source") == "main_model_selected_frame"
            or not isinstance(qbc.get("member_count"), int)
            or qbc["member_count"] <= 1
            or not pool
            or not isinstance(pool[0], dict)
        ):
            continue
        mean = pool[0].get("energy_mean_per_atom")
        energy = outputs.get("energy_per_atom")
        total = outputs.get("energy")
        record_energy = record.get("energy")
        values = (mean, energy, total, record_energy)
        if (
            all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                for v in values
            )
            and math.isclose(mean, energy, rel_tol=1e-10, abs_tol=1e-10)
            and math.isclose(total, record_energy, rel_tol=1e-10, abs_tol=1e-10)
        ):
            matched.append(entry["record_id"])
    return matched


def assert_snapshot_energy_consistency(state, snapshot):
    matched = legacy_mc_mean_record_ids(state, snapshot)
    if matched:
        raise ValueError(
            f"当前相图包含{len(matched)}条MC委员会均值能量，与主模型能量口径不一致；请先统一能量并重建相图"
        )

"""Explain saved per-site magnetic evidence without rereading OUTCAR or running models."""
import json

MAGNETIC_FIELDS = ("model_version", "task_id", "structure_id", "frame_index", "atom_index",
                   "element", "moment_mu_B", "moment_kind", "spin_assessment",
                   "expected_abs_min_mu_B", "expected_abs_max_mu_B", "reason")


def magnetic_rows(records):
    rows = []
    for record in sorted(records, key=lambda row: str(row.get("task_id"))):
        check = record.get("magnetic_check") or {}
        raw = record.get("magnetic_moments") or check
        elements = raw.get("elements") or check.get("elements") or []
        moments = raw.get("moments") or []
        good = {r["atom_index"]: r for r in check.get("reasonable_atoms") or []}
        bad = {r["atom_index"]: r for r in check.get("anomalous_atoms") or []}
        applies = (record.get("spin_state_check") or {}).get("applies", False)
        for index, element in enumerate(elements):
            evidence = good.get(index) or bad.get(index) or {}
            assessment = ("reasonable" if index in good else "out_of_range" if index in bad else
                          "unknown" if applies and element in {"Fe", "Mn"} else "not_checked")
            moment = moments[index] if index < len(moments) else None
            interval = evidence.get("expected_abs_range") or (check.get("expected_abs_ranges") or {}).get(element) or [None, None]
            rows.append({"model_version": record.get("model_version"), "task_id": record.get("task_id"),
                "structure_id": record.get("structure_id"), "frame_index": raw.get("final_frame_index"),
                "atom_index": index, "element": element,
                "moment_mu_B": json.dumps(moment) if isinstance(moment, list) else moment,
                "moment_kind": "vector_xyz" if raw.get("noncollinear") else "scalar",
                "spin_assessment": assessment, "expected_abs_min_mu_B": interval[0],
                "expected_abs_max_mu_B": interval[1],
                "reason": check.get("reason") or check.get("error") if assessment == "unknown" else None})
    return rows


def magnetic_chat_summary(state, *, detail_limit=4, task_ids=None, verbose=True):
    """Compact normal/abnormal explanation, with bounded task details."""
    saved = {r.get("task_id"): dict(r) for r in state.get("dft_dataset_records") or []}
    for task in state.get("tasks") or []:
        if task.get("stage") in {"dft_single_point", "dft_relax"} and task.get("status") in {"completed", "failed", "timeout"}:
            saved.setdefault(task.get("task_id"), {**task, **(task.get("outputs") or {})})
    records = [r for key, r in saved.items() if task_ids is None or key in task_ids]
    if not records:
        return ""
    checked = [r for r in records if (r.get("spin_state_check") or {}).get("applies")]
    missing = [r for r in records if not ((r.get("magnetic_moments") or r.get("magnetic_check") or {}).get("moments"))]
    unassessed = [r for r in records if not r.get("spin_state_check")]
    skipped = sum((r.get("spin_state_check") or {}).get("status") == "not_applicable" for r in records)
    counts = {name: sum((r.get("spin_state_check") or {}).get("status") == name for r in checked)
              for name in ("passed", "rejected", "unknown")}
    if not verbose:
        text = f"磁矩：合格 {counts['passed']}、异常 {counts['rejected']}、未知 {counts['unknown']}、不适用 {skipped}。"
        if missing or unassessed:
            text += f"缺少磁矩 {len(missing)} 个、未评估 {len(unassessed)} 个；缺失证据需重新提取。"
        return text
    lines = [f"磁矩：已回收 {len(records)} 个结果，缺少磁矩 {len(missing)} 个、未评估 {len(unassessed)} 个。",
             f"层状 Fe/Mn 自旋：符合标准 {counts['passed']}、异常 {counts['rejected']}、未知 {counts['unknown']}；不适用 {skipped} 个。",
             "预期 |磁矩|：Fe 3.5–4.5、Mn 3.0–4.0 μB；按原子逐一检查，不是严格基态证明。"]
    normal = [a for r in checked for a in (r.get("magnetic_check") or {}).get("reasonable_atoms") or []]
    details = []
    for element in ("Fe", "Mn"):
        values = [a["absolute_moment"] for a in normal if a["element"] == element]
        if values:
            details.append(f"{element} {len(values)} 个原子（|m|={min(values):.3g}–{max(values):.3g} μB）")
    if details:
        lines.append("符合范围：" + "；".join(details) + "。")
    problematic = [r for r in checked if (r.get("spin_state_check") or {}).get("status") != "passed"]
    for record in problematic[-detail_limit:]:
        check = record.get("magnetic_check") or {}
        atoms = check.get("anomalous_atoms") or []
        detail = "、".join(f"{a['element']}[{a['atom_index']}]={a['moment']:.3g}" for a in atoms[:3])
        lines.append(f"- {record.get('task_id')}：{detail + ' μB' if detail else check.get('reason') or '证据不完整'}。")
    scopes = [r.get("round_scope") for r in records if r.get("round_scope")]
    paths = list(dict.fromkeys(r.get("magnetic_csv_path") for r in (state.get("dft_result_exports") or {}).values()
                 if r.get("magnetic_csv_path") and r.get("round_scope") in scopes))
    if paths:
        lines.append("逐原子清单：" + "、".join(f"`{path}`" for path in paths[:3]) + (f"（共 {len(paths)} 轮）" if len(paths) > 3 else ""))
    if missing:
        lines.append("层状 Fe/Mn 的缺失磁矩不能由能量/受力恢复，需在超算重新提取并回传同任务结果；补充检查不重复记账。")
    lines.append("层状 Fe/Mn 数据仅通过检查后参与相图、微调和误差评估；原始结果保留。")
    return "\n".join(lines)

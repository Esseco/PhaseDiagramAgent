"""Plot verified same-frame per-epoch MAE without mixing K-fold evidence."""

import csv
import io
import json
import math
from pathlib import Path
import re

from phase_agent.analysis.feedback.export_dft_products import _publish_bytes
from phase_agent.analysis.feedback.versioned_products import validate_bundle


def publish_round_error_trend(state, directory):
    root = Path(directory).resolve()
    if root.name != "analysis_outputs" or not root.is_dir():
        raise ValueError("registered analysis directory required")
    from collections import defaultdict
    from phase_agent.analysis.feedback.dft_comparison_tables import (
        build_comparison_tables,
        comparison_metrics,
    )
    from phase_agent.evidence.model_epoch import model_epoch

    from phase_agent.evidence.dft_extrapolation import failed_dft_task_ids

    failed_ids = failed_dft_task_ids(state)
    accepted = defaultdict(list)
    for history in state.get("qbc_selection_history") or []:
        for item in history.get("accepted") or []:
            if item.get("task_key") and item.get("candidate_id"):
                accepted[item["task_key"]].append(item)
    parent_tasks = set()
    for batch in (state.get("parent_first_recovery_batches") or {}).values():
        digest = batch.get("sampling_result_sha256") or ""
        if len(digest) == 64 and batch.get("model_version"):
            parent_tasks.update(
                (task_id, batch["model_version"]) for task_id in batch.get("dft_task_ids") or []
            )
    tasks = {task.get("task_id"): task for task in state.get("tasks") or [] if task.get("task_id")}
    comparisons = state.get("dft_mlip_comparisons") or []
    groups = defaultdict(list)
    excluded = defaultdict(int)
    excluded_failed = defaultdict(list)
    for record in state.get("dft_dataset_records") or []:
        model = record.get("model_version")
        epoch = model_epoch(state, model)
        if not epoch:
            continue
        if record.get("task_id") in failed_ids:
            excluded_failed[(epoch, model)].append(record)
            continue
        task = tasks.get(record.get("task_id")) or {}
        if task.get("model_version") != model or task.get("task_key") != record.get("task_key"):
            excluded[(epoch, model)] += 1
            continue
        key = record.get("task_key")
        ordinary = any(
            item.get("candidate_id") in {record.get("structure_id"), task.get("object_id")}
            and (
                not item.get("input_structure_sha256")
                or item.get("input_structure_sha256") == task.get("input_structure_sha256")
            )
            for item in accepted.get(key, [])
        )
        parent_first = (record.get("task_id"), model) in parent_tasks
        if ordinary or parent_first:
            groups[(epoch, model)].append(record)
        else:
            excluded[(epoch, model)] += 1
    epochs = sorted(
        (p.name for p in root.iterdir() if p.is_dir() and re.fullmatch(r"epoch[0-9]+", p.name)),
        key=lambda name: int(name[5:]),
    )
    rows = []
    for epoch in epochs:
        versions = sorted(
            {model for ep, model in groups if ep == epoch}
            | {model for ep, model in excluded if ep == epoch}
            | {model for ep, model in excluded_failed if ep == epoch}
        )
        if not versions:
            rows.append(
                {"epoch": epoch, "status": "unavailable", "reason": "没有已登记的新选 DFT 数据"}
            )
            continue
        for model in versions:
            records = groups[(epoch, model)]
            row = {
                "epoch": epoch,
                "model_version": model,
                "status": "unavailable",
                "selected_records": len(records),
                "selection_unverified": excluded[(epoch, model)],
                "excluded_extrapolation_failed": len(excluded_failed[(epoch, model)]),
            }
            failed_records = excluded_failed[(epoch, model)]
            if failed_records:
                failed_energy, _ = build_comparison_tables(
                    failed_records, comparisons, round_name="excluded-extrapolation-failed"
                )
                row["excluded_failed_previously_paired"] = sum(
                    item.get("comparison_status") == "completed" for item in failed_energy
                )
            else:
                row["excluded_failed_previously_paired"] = 0
            if not records:
                row["reason"] = "DFT 记录没有可核实的新选来源"
                rows.append(row)
                continue
            try:
                ids = [record.get("task_id") for record in records]
                if len(ids) != len(set(ids)):
                    raise ValueError("选点 DFT 任务记录重复")
                id_set = set(ids)
                relevant = [item for item in comparisons if item.get("task_id") in id_set]
                if len({item.get("task_id") for item in relevant}) != len(relevant):
                    raise ValueError("同帧模型比较记录重复")
                energies, forces = build_comparison_tables(
                    records, relevant, round_name="selected-epoch"
                )
                metrics = comparison_metrics(
                    energies, forces, {"model_version": model, "epoch": epoch}
                )
                e, f = float(metrics["energy_per_atom"]["mae"]), float(metrics["forces"]["mae"])
                structures, components = (
                    int(metrics["matched_structures"]),
                    int(metrics["forces"]["components"]),
                )
                if (
                    not all(math.isfinite(x) and x >= 0 for x in (e, f))
                    or structures < 1
                    or components < 1
                ):
                    raise ValueError("没有有效的选点同帧能量和受力配对")
                row.update(
                    status="completed",
                    energy_mae_eV_per_atom=e,
                    force_mae_eV_per_A=f,
                    structures=structures,
                    force_components=components,
                    excluded_unpaired=len(metrics["not_evaluated"]),
                    is_active_model=model == state.get("active_model_version"),
                )
            except (ValueError, KeyError, TypeError) as error:
                row["reason"] = str(error)
            rows.append(row)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib import rc_context

    latest = root / "latest"
    fields = (
        "epoch",
        "status",
        "model_version",
        "selected_records",
        "selection_unverified",
        "excluded_extrapolation_failed",
        "excluded_failed_previously_paired",
        "excluded_unpaired",
        "is_active_model",
        "energy_mae_eV_per_atom",
        "force_mae_eV_per_A",
        "structures",
        "force_components",
        "reason",
    )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    _publish_bytes(latest / "Round_Error_MAE_by_epoch.csv", buffer.getvalue().encode("utf-8-sig"))
    for kind, key, label, sample in (
        ("E", "energy_mae_eV_per_atom", "Energy MAE (meV/atom)", "structures"),
        ("F", "force_mae_eV_per_A", "Force MAE (meV/Å)", "force_components"),
    ):
        with rc_context({"font.family": "sans-serif", "font.size": 10}):
            fig = Figure(figsize=(7, 4.8))
            FigureCanvasAgg(fig)
            ax = fig.subplots()
            xs = [int(r["epoch"][5:]) for r in rows]
            ys = [1000 * r[key] if r["status"] == "completed" else float("nan") for r in rows]
            if any(r["status"] == "completed" for r in rows):
                if len(set(xs)) == len(xs):
                    ax.plot(xs, ys, marker="o", linewidth=1.5)
                else:
                    ax.scatter(xs, ys, marker="o")
                for x, y, r in zip(xs, ys, rows):
                    if r["status"] == "completed":
                        ax.annotate(
                            f"n={r[sample]}",
                            (x, y),
                            xytext=(4, 6),
                            textcoords="offset points",
                            fontsize=8,
                        )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No verified same-frame results",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            ticks = sorted(set(xs))
            ax.set_xticks(ticks, [f"epoch{x}" for x in ticks])
            ax.set_ylabel(label)
            ax.set_xlabel("Model epoch")
            ax.set_title("Round DFT vs original-round MLIP")
            fig.text(
                0.5,
                0.02,
                "Datasets can differ across epochs; not an independent test",
                ha="center",
                fontsize=8,
            )
            fig.tight_layout(rect=(0, 0.06, 1, 1))
            for extension in ("png",):
                out = io.BytesIO()
                fig.savefig(out, format=extension, dpi=200)
                _publish_bytes(
                    latest / f"Round_Error_{kind}_MAE_by_epoch.{extension}", out.getvalue()
                )
            fig.clear()
    result = {
        "evaluation_type": "original_round_model_same_frame",
        "interpretation": "仅包含普通选点或已登记 parent-first 选点；已确认外推失败、未核实选点来源及未配对记录排除；失败记录数包含原本未配对的任务，另列原先可配对结构数。各 epoch 结构集合不同；非固定测试集或独立测试。",
        "rows": rows,
    }
    _publish_bytes(
        latest / "Round_Error_MAE_by_epoch.json",
        (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf8"),
    )
    return result

"""Discover saved products without calculating or scanning raw simulation data."""

from pathlib import Path
import os
import re
from phase_agent.analysis.state.model_epoch import model_epoch


CATALOG_FIELDS = (
    "epoch",
    "model_version",
    "search_group_index",
    "round",
    "product",
    "path",
    "exists",
)


SUBMISSION_PRODUCTS = {
    "remote_training_inputs",
    "training_returned_results",
    "submission_files",
    "returned_results",
}


def _relative_target(path, root):
    try:
        return Path(os.path.relpath(Path(path).resolve(), root.resolve())).as_posix()
    except ValueError:
        return Path(path).as_posix()


def _project_overview(root, state):
    from phase_agent.tools.step_runner.build_status_summary import build_status_summary

    summary = build_status_summary(state)
    epoch = model_epoch(state, summary["model_version"]) or "未登记"
    counts = summary["task_counts"]
    labels = {"pending": "待回传/核对", "completed": "已完成", "failed": "失败待核对"}
    tasks = "；".join(f"{labels.get(key, key)} {value}" for key, value in sorted(counts.items()))
    structures = [
        f"[{label}](<{_relative_target(root.parent / relative, root)}>)"
        for label, relative in (
            ("母结构", "structures/reference_structures"),
            ("候选结构", "structures/candidate_structures"),
        )
        if (root.parent / relative).is_dir()
    ]
    return [
        "## 项目总览",
        "",
        f"项目：{root.parent.name}；当前轮：{epoch}；待审批方案：{summary['pending_approvals']}。",
        f"本地任务共 {sum(counts.values())} 项：{tasks or '尚无任务'}。",
        "本地待回传记录不代表远端未提交或仍在运行。",
        "结构入口：" + (" · ".join(structures) if structures else "尚未初始化"),
        "",
    ]


def catalog_markdown(rows, root, state=None):
    root = Path(root)
    indexes = sorted(
        {row["submission_index_path"] for row in rows if row.get("submission_index_path")}
    ) or [str(root.parent / "submissions" / "README.md")]
    lines = [
        "# 输出文件索引",
        "",
        "仅索引已登记的分析产物，不分析科学数据。缺失项标为缺失，不视为完成。",
        "",
        *[f"- 提交与原始回传：[提交索引](<{_relative_target(path, root)}>)" for path in indexes],
        (
            "- [跨 epoch summary](round_summary.csv)：模型与 DFT 轮次汇总。"
            if (root / "round_summary.csv").is_file()
            else "- `round_summary.csv`：尚未生成。"
        ),
        "- 各 epoch 的 `round_metrics.csv`：本模型各 DFT 轮能量与受力误差。",
        "",
        "## 当前模型分析",
        "",
    ]
    if state is not None:
        lines[2:2] = _project_overview(root, state)
    derived = [row for row in rows if row["product"] not in SUBMISSION_PRODUCTS]

    def append_rows(selected):
        previous = None
        for row in selected:
            label = f"{row['epoch']} · {row['model_version']}"
            if label != previous:
                lines.extend([f"### {label}", ""])
                previous = label
            target = Path(row["path"])
            group = (
                f"Search-group-{int(row['search_group_index']):04d} / "
                if row["search_group_index"]
                else ""
            )
            product = {
                "relax_mc_report": "相图审阅报告",
                "relax_mc_plot": "相图与质量图片",
            }.get(row["product"], row["product"]).replace("|", "\\|")
            status = "存在" if row["exists"] else "缺失"
            lines.append(
                f"- {group}{row['round']} · {product}（{status}）："
                f"[{target.name}](<{_relative_target(target, root)}>)"
            )

    current = [row for row in derived if row.get("current_model")]
    if current:
        append_rows(current)
    else:
        lines.append("当前模型尚无已登记的分析产物。")
    historical = [row for row in derived if not row.get("current_model")]
    if historical:
        lines.extend(["", "<details>", "<summary>历史模型分析</summary>", ""])
        append_rows(historical)
        lines.extend(["", "</details>"])
    return "\n".join(lines) + "\n"


def output_catalog(state):
    rows = []

    def add(version, group, round_name, product, path):
        if path:
            rows.append(
                {
                    "epoch": model_epoch(state, version) or "unrecorded",
                    "model_version": version,
                    "search_group_index": group,
                    "round": round_name,
                    "product": product,
                    "path": str(path),
                    "exists": Path(path).exists(),
                    "current_model": bool(state.get("active_model_version"))
                    and version == state["active_model_version"],
                }
            )
            if product in SUBMISSION_PRODUCTS:
                epoch = next(
                    (
                        parent
                        for parent in Path(path).parents
                        if re.fullmatch(r"epoch\d+(?:_.+)?", parent.name)
                    ),
                    None,
                )
                if epoch:
                    rows[-1]["submission_index_path"] = str(epoch.parent / "README.md")

    for method, snapshot in (state.get("phase_diagrams") or {}).items():
        version = snapshot.get("model_version") or state.get("active_model_version")
        stored = snapshot.get("csv_path")
        add(version, "", "current", f"phase_diagram_{method}", stored)
        # Existing images in this model/method folder can cover several saved rounds.
        # Link the directory; never claim that an arbitrary image is the latest hull.
        if stored:
            images = Path(stored).parent / "plots"
            if images.is_dir():
                add(version, "", "已保存相图", f"{method} 相图图片目录", images)
    for export in (state.get("dft_result_exports") or {}).values():
        scope = export.get("round_scope") or {}
        version, group = scope.get("model_version"), scope.get("search_group_index")
        round_name = Path(export["directory"]).name
        for field, product in (
            ("energy_csv_path", "energy_parity"),
            ("force_csv_path", "force_components_parity"),
            ("metrics_csv_path", "energy_force_MAE_RMSE"),
            ("magnetic_csv_path", "magnetic_moments"),
        ):
            add(version, group, round_name, product, export.get(field))
        for name, path in (export.get("magnetic_plots", {}).get("files") or {}).items():
            add(version, group, round_name, f"magnetic_distribution_{name}", path)
        for category in ("comparisons/plots", "training"):
            path = Path(export["directory"]) / category
            if path.exists():
                add(version, group, round_name, category, path)
    for bundle_id, batch in (state.get("relax_mc_batches") or {}).items():
        analysis = batch.get("phase_analysis") or {}
        version = batch.get("model_version")
        label = batch.get("package_label") or bundle_id
        add(version, "", label, "relax_mc_report", analysis.get("markdown_path"))
        for path in analysis.get("plot_paths") or []:
            add(version, "", label, "relax_mc_plot", path)
    for job in (state.get("remote_finetune_jobs") or {}).values():
        if job.get("status") in {"abandoned", "superseded", "cancelled"}:
            continue
        training = Path(job["directory"])
        add(
            job.get("original_model_version"),
            "",
            training.name,
            "remote_training_inputs",
            training / "inputs" if (training / "inputs").is_dir() else training,
        )
        add(
            job.get("original_model_version"),
            "",
            training.name,
            "training_returned_results",
            training / "results",
        )
    stages = set()
    for batch in state.get("slurm_batches") or []:
        if not batch.get("upload_directory"):
            continue
        # One entry for the full stage directory, not one line per task/batch.
        stage = Path(batch["upload_directory"]).parent
        inputs = stage
        if stage.name == "inputs":
            stage = stage.parent
        key = (batch.get("model_version"), batch.get("search_group_index"), str(stage))
        if key in stages:
            continue
        stages.add(key)
        add(key[0], key[1], stage.name, "submission_files", inputs)
        add(key[0], key[1], stage.name, "returned_results", stage / "results")
    return sorted(rows, key=lambda row: tuple(str(row[key]) for key in CATALOG_FIELDS[:-1]))

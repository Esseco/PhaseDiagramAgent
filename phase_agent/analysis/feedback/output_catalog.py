"""Discover saved products without calculating or scanning raw simulation data."""

from pathlib import Path
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


def catalog_markdown(rows, root):
    lines = [
        "# 输出文件索引",
        "",
        "由 Agent 维护；仅索引已有文件，不分析科学数据。缺失项标为缺失，不视为完成。",
        "",
        "- `round_summary.csv`：跨 epoch 流程总览。",
        "- 各 epoch 的 `round_metrics.csv`：本模型各 DFT 轮能量与受力误差。",
        "",
    ]
    current = None
    for row in rows:
        label = f"{row['epoch']} · {row['model_version']}"
        if label != current:
            lines.extend([f"## {label}", ""])
            current = label
        target = Path(row["path"])
        try:
            target_text = target.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            target_text = target.as_posix()
        group = (
            f"Search-group-{int(row['search_group_index']):04d} / "
            if row["search_group_index"]
            else ""
        )
        product = row["product"].replace("|", "\\|")
        status = "存在" if row["exists"] else "缺失"
        lines.append(
            f"- {group}{row['round']} · {product}（{status}）：[{target.name}](<{target_text}>)"
        )
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
                }
            )

    for method, snapshot in (state.get("phase_diagrams") or {}).items():
        version = snapshot.get("model_version") or state.get("active_model_version")
        add(version, "", "current", f"phase_diagram_{method}", snapshot.get("csv_path"))
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
        for category in ("comparisons/plots", "training"):
            path = Path(export["directory"]) / category
            if path.exists():
                add(version, group, round_name, category, path)
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

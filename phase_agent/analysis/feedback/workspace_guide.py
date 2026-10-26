"""Publish navigation only, not computed scientific results."""

import json
from pathlib import Path


def publish_workspace_guide(root):
    from phase_agent.analysis.feedback.export_dft_products import _publish_bytes

    root = Path(root)
    text = _read_template("workspace.md")
    if not (root / "submissions/submission_index.json").is_file():
        text = text.replace(
            "Agent 先读 [机器索引](submissions/submission_index.json)",
            "`submissions/submission_index.json`（登记任务后生成）",
        )
    for label, relative in (
        ("母结构", "structures/reference_structures"),
        ("候选结构", "structures/candidate_structures"),
    ):
        if not (root / relative).is_dir():
            text = text.replace(f"[{label}]({relative}/)", f"{label}（初始化后生成）")
    if not (root / "analysis_outputs/output_index.md").is_file():
        text = text.replace(
            "[结果索引](analysis_outputs/output_index.md)",
            "analysis_outputs/（首次产生结果后生成索引）",
        )
    if (root / "runtime/state.json").is_file() and not (
        root / "workflow_state/state.json"
    ).is_file():
        # Existing projects keep their explicit layout until authorized migration.
        from phase_agent.configuration.session.workspace_directory_names import DIRECTORY_RENAMES

        for old, new in sorted(DIRECTORY_RENAMES.items(), key=lambda item: -len(item[1])):
            text = text.replace(new, old)
    _publish_bytes(root / "README.md", text.encode("utf-8"))
    for directory in ("interaction_records", "submissions", "analysis_outputs"):
        target = root / directory / "README.md"
        if not target.exists():
            _publish_bytes(target, _read_template(directory + ".md").encode("utf-8"))
    if not (root / "runtime/state.json").is_file():
        details = _read_template("file_layout.md")
        if not (root / "analysis_outputs/output_index.md").is_file():
            details = details.replace(
                "[文件索引](../analysis_outputs/output_index.md)",
                "analysis_outputs/output_index.md（产生结果后生成）",
            )
        _publish_bytes(root / "documentation/FILE_LAYOUT.md", details.encode("utf-8"))


def _append_epoch_manifest_summary(text, directory):
    manifest_path = Path(directory) / "layout_manifest.json"
    if not manifest_path.is_file():
        return text
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    labels = {
        "calculation": "计算层",
        "delivery": "交付层",
        "evidence": "证据层",
        "quarantine": "隔离层",
        "unclassified": "未分类（需核对）",
    }
    lines = [
        "",
        "## 本轮直接子目录",
        "",
        "目录职责由 `layout_manifest.json` 固定；未分类目录不作为提交入口。",
        "",
        "| 目录 | 层 | 登记 |",
        "| --- | --- | --- |",
    ]
    for row in manifest.get("children") or []:
        registered = (
            "历史旧包（隔离）"
            if row.get("legacy")
            else "已登记"
            if row.get("registered")
            else "固定命名空间"
        )
        lines.append(
            f"| `{row['name']}/` | {labels.get(row.get('lane'), row.get('lane'))} | {registered} |"
        )
        for child in row.get("children") or []:
            child_registered = (
                "历史旧包（隔离）"
                if child.get("legacy")
                else "已登记"
                if child.get("registered")
                else "固定命名空间"
            )
            lines.append(
                f"| `{row['name']}/{child['name']}/` | "
                f"{labels.get(child.get('lane'), child.get('lane'))} | {child_registered} |"
            )
            for package in child.get("children") or []:
                package_registered = (
                    "历史旧包（隔离）"
                    if package.get("legacy")
                    else "已登记"
                    if package.get("registered")
                    else "固定命名空间"
                )
                lines.append(
                    f"| `{row['name']}/{child['name']}/{package['name']}/` | "
                    f"{labels.get(package.get('lane'), package.get('lane'))} | {package_registered} |"
                )
    if manifest.get("unclassified"):
        lines.extend(
            [
                "",
                "未分类目录：" + "、".join(f"`{name}/`" for name in manifest["unclassified"]),
            ]
        )
    return text.rstrip() + "\n" + "\n".join(lines) + "\n"


def _read_template(name):
    """Load documentation resources; links resolve in the generated workspace."""
    return (Path(__file__).with_name("templates") / name).read_text(encoding="utf-8")


def publish_epoch_guides(root, state):
    """Describe only existing recorded epochs; never create a scientific result."""
    from phase_agent.analysis.feedback.export_dft_products import _publish_bytes

    root = Path(root)
    for version, number in ((state.get("upload_layout") or {}).get("model_rounds") or {}).items():
        epoch = f"epoch{int(number) - 1}"
        directory = root / epoch
        if directory.is_dir():
            text = f"# {epoch}\n\n模型版本：`{version}`。任务和批次编号保持原登记。\n\n"
            template = (
                root.name
                if root.name in {"submissions", "analysis_outputs", "interaction_records"}
                else "analysis_outputs"
            )
            text += _read_template(template + ".md").split("\n", 1)[1]
            if template == "submissions":
                text = text.replace(
                    "`submission_index.json`", "`../submission_index.json`；`layout_manifest.json`"
                )
                text = text.replace("`previously_used/`", "`../previously_used/<epoch>/`")
                text = text.replace(
                    "`epochN/evidence/storage_archives/`",
                    "`evidence/storage_archives/`（本轮存在时）",
                )
                text = _append_epoch_manifest_summary(text, directory)
            _publish_bytes(directory / "README.md", text.encode("utf-8"))

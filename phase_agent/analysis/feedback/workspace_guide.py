"""Publish navigation only, not computed scientific results."""

from pathlib import Path


def publish_workspace_guide(root):
    from phase_agent.analysis.feedback.export_dft_products import _publish_bytes

    root = Path(root)
    text = _read_template("workspace.md")
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
            _publish_bytes(directory / "README.md", text.encode("utf-8"))

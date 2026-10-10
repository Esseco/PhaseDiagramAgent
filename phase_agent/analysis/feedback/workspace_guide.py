"""Publish navigation only, not computed scientific results."""

from pathlib import Path


def publish_workspace_guide(root):
    from phase_agent.analysis.feedback.export_dft_products import _publish_bytes

    root = Path(root)
    text = _read_template("workspace.md")
    if (root / "runtime/state.json").is_file() and not (
        root / "workflow_state/state.json"
    ).is_file():
        # Existing projects keep their explicit layout until authorized migration.
        from phase_agent.configuration.session.workspace_directory_names import DIRECTORY_RENAMES

        for old, new in sorted(DIRECTORY_RENAMES.items(), key=lambda item: -len(item[1])):
            text = text.replace(new, old)
    _publish_bytes(root / "README.md", text.encode("utf-8"))
    if not (root / "runtime/state.json").is_file():
        details = _read_template("file_layout.md")
        _publish_bytes(root / "documentation/FILE_LAYOUT.md", details.encode("utf-8"))


def _read_template(name):
    """Load documentation resources; links resolve in the generated workspace."""
    return (Path(__file__).with_name("templates") / name).read_text(encoding="utf-8")

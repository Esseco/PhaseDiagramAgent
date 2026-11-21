"""Publish the pending refresh status before the guarded state commit."""

from pathlib import Path


def commit_reopen(updated, state_path, expected_identity, *, epoch):
    from phase_agent.analysis.feedback.export_dft_products import _publish_bytes
    from phase_agent.analysis.feedback.versioned_products import publication_status
    from phase_agent.persistence.state.read_versions import read_identity
    from phase_agent.tools.workflows.lifecycle_support import _save_runner_state

    state_path = Path(state_path).resolve()
    root = Path(updated["analysis_output_directory"]).resolve()
    if root.name != "analysis_outputs" or root.parent != state_path.parent.parent:
        raise ValueError("reopen_publication_root_invalid")
    if not isinstance(epoch, str) or not epoch.startswith("epoch") or not epoch[5:].isdigit():
        raise ValueError("reopen_epoch_invalid")
    paths = [root / "latest" / "phase_diagram_status.json", root / epoch / "phase_diagram" / "latest_status.json"]
    previous = {p: p.read_bytes() if p.is_file() else None for p in paths}
    try:
        publication_status(root, "phase_diagram", updated["active_model_version"], epoch,
                           status="not_ready", reason="completed_refresh_superseded")
        _save_runner_state(updated, state_path, expected_identity=expected_identity)
    except BaseException:
        # If the state was already committed, keeping not_ready is safer than
        # making an old image look current after a post-commit failure.
        if read_identity(state_path, strong=True) == expected_identity:
            for path, value in previous.items():
                if value is None:
                    path.unlink(missing_ok=True)
                else:
                    _publish_bytes(path, value)
        raise

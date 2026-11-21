"""Version-bound, minimal state for rendering the existing phase diagram."""

_PLOT_FIELDS = (
    "active_model_version",
    "active_model",
    "model_registry",
    "extrapolation_recovery_cycle",
    "model_refresh",
    "analysis_output_directory",
    "confirmed_config",
    "phase_diagrams",
    "phase_diagrams_by_model",
    "upload_layout",
    "relax_mc_batches",
    "tasks",
    "phase_records",
    "phase_quality_archived_records",
    "phase_quality_policy",
    "phase_quality_decisions",
    "phase_quality_exclusions",
    "phase_quality_restorations",
    "phase_boundary_cleanup",
)


def read_plot_state(path):
    """Select only plot and quality inputs; fall back if the index is unsafe."""
    from phase_agent.persistence.state.field_index import IndexUnavailable, open_indexed_state
    from phase_agent.runtime.turn_process import has_reusable_initial_state
    from phase_agent.tools.step_runner.file_protocol import read_json

    indexed = None if has_reusable_initial_state(path) else open_indexed_state(path)
    if indexed is not None:
        try:
            projection = {key: indexed[key] for key in _PLOT_FIELDS if key in indexed.fields}
            indexed.verify(remember=True)
            return projection
        except (IndexUnavailable, OSError, ValueError, TypeError, KeyError):
            pass
    return read_json(path, {}) or {}

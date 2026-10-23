"""Recognize unchanged confirmed files without reopening a configuration review."""

from phase_agent.analysis.state.project_authority import configuration_changes
from phase_agent.configuration.session.load_editable_config_json import (
    load_editable_config_json,
)
from phase_agent.configuration.session.resolve_phase_reference_directory import (
    resolve_phase_reference_directory,
)
from phase_agent.configuration.session.replace_config_from_json import (
    mother_structure_digest,
)
from phase_agent.runtime.editable_config_baseline import editable_comparison_baseline


def confirmed_file_unchanged(handler, session):
    snapshot = (session.get("confirmed_snapshot") or {}).get("config")
    source = handler.editable_config_path
    if snapshot is None or source is None:
        return False
    try:
        config = load_editable_config_json(source, snapshot)
        config = resolve_phase_reference_directory(config, base_directory=handler.base_directory)
        expected = editable_comparison_baseline(snapshot, source, handler.base_directory)
        if configuration_changes(expected, config):
            return False
        known = session.get("agent_reviewed_mother_digest") or session.get(
            "last_imported_mother_digest"
        )
        baseline = session.get("confirmed_file_baseline") or {}
        if not known and baseline.get("config_version") == (
            session.get("confirmed_snapshot") or {}
        ).get("config_version"):
            known = baseline.get("mother_digest")
        return bool(known and known == mother_structure_digest(config))
    except (OSError, ValueError, TypeError, KeyError):
        return False

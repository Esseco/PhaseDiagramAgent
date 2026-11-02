"""Bind existing read views without composing or migrating a scientific runtime."""

from dataclasses import dataclass
from pathlib import Path

from phase_agent.runtime.local_agent_control import LocalAgentControl
from phase_agent.runtime.saved_analysis_project import SavedAnalysisView
from phase_agent.runtime.runtime_config_io import (
    _load_json_object,
    _resolve_path,
    runtime_base_directory,
)


@dataclass
class SavedReadOnlyView(SavedAnalysisView):
    config_session_path: Path | None = None
    base_directory: Path | None = None
    editable_config_path: Path | None = None


class SavedReadOnlyProject:
    """Read status, approved/pending memory and approvals; no model or workflow."""

    def __init__(self, runtime_config):
        config_path = Path(runtime_config).resolve()
        settings = _load_json_object(config_path, "read-only runtime config")
        base = runtime_base_directory(config_path, settings)
        if not settings.get("state_path"):
            raise ValueError("Read-only access requires an explicit saved state path")
        state_path = _resolve_path(settings["state_path"], base)
        if not state_path.is_file():
            raise FileNotFoundError(f"Saved state does not exist: {state_path}")
        session = settings.get("config_session_path")
        editable = settings.get("editable_config_draft_path")
        self.control = LocalAgentControl(
            SavedReadOnlyView(
                state_path=state_path,
                config_session_path=_resolve_path(session, base) if session else None,
                base_directory=base,
                editable_config_path=_resolve_path(editable, base) if editable else None,
            )
        )

    def pending(self):
        return self.control.pending()

"""Bind the existing analysis control to saved files without initializing a scientific runtime."""

from dataclasses import dataclass, field
from pathlib import Path
import threading

from phase_agent.runtime.local_agent_control import LocalAgentControl
from phase_agent.runtime.runtime_config_io import (
    _load_json_object,
    _resolve_path,
    runtime_base_directory,
)


@dataclass
class SavedAnalysisView:
    state_path: Path
    lock: object = field(default_factory=threading.RLock)
    analysis_output_directory: Path | None = None


class SavedAnalysisProject:
    """A restricted analysis control: no model clients, scientific state changes or workflow."""

    def __init__(self, runtime_config):
        config_path = Path(runtime_config).resolve()
        settings = _load_json_object(config_path, "analysis runtime config")
        base = runtime_base_directory(config_path, settings)
        if not settings.get("state_path") or not settings.get("phase_diagram_directory"):
            raise ValueError("Saved analysis requires explicit state and output directory bindings")
        state_path = _resolve_path(settings["state_path"], base)
        state = _load_json_object(state_path, "saved analysis state")
        registered = state.get("analysis_output_directory")
        configured = _resolve_path(settings["phase_diagram_directory"], base)
        if registered:
            if Path(registered).resolve() != configured:
                raise ValueError("Registered analysis output does not match project runtime config")
        elif not (
            state_path == (base / "workflow_state" / "state.json").resolve()
            and configured == (base / "analysis_outputs").resolve()
        ):
            raise ValueError("Unregistered analysis output requires standard project-local paths")
        # Legacy projects can use the validated runtime directory for this
        # read-only analysis session. Never persist it as scientific state.
        self.control = LocalAgentControl(SavedAnalysisView(state_path, analysis_output_directory=configured))

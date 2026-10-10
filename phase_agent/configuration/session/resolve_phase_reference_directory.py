"""Expand one mother-structure directory into per-phase file references."""

from copy import deepcopy
from pathlib import Path
from phase_agent.science.structures.boundary_utils import allowed_phases


def resolve_phase_reference_directory(config: dict, *, base_directory=None) -> dict:
    """Resolve ``<phase>.vasp`` files from the configured shared directory.

    Explicit entries in ``system.phase_references`` take precedence. The
    returned config is a copy; callers decide whether and when to save it.
    """
    resolved = deepcopy(config)
    system = resolved.get("system")
    if not isinstance(system, dict):
        return resolved
    directory = system.get("phase_reference_directory")
    if directory in (None, ""):
        return resolved
    if not isinstance(directory, str):
        raise ValueError("system.phase_reference_directory 必须是目录路径字符串")

    root = Path(directory).expanduser()
    if not root.is_absolute() and base_directory is not None:
        root = Path(base_directory) / root
    boundary = system.get("boundary") or {}
    phases = boundary.get("P") if isinstance(boundary, dict) else None
    if isinstance(boundary, dict) and "P" in boundary and not phases:
        # Missing scientific choices must not prevent a configuration conversation.
        # Do not fall back to template phases or manufacture reference paths.
        return resolved
    if not phases:
        phases = (system.get("constraints") or {}).get("phases") or []
    phases = sorted(allowed_phases(phases)) if phases else []
    if not phases or any(not phase.strip() for phase in phases):
        raise ValueError("boundary.P 必须包含相名，才能从公共目录推导母结构文件")

    explicit = system.get("phase_references") or {}
    if not isinstance(explicit, dict):
        raise ValueError("system.phase_references 必须是 JSON 对象")
    references = {phase: str(root / f"{phase}.vasp") for phase in phases}
    references.update(explicit)
    system["phase_references"] = references
    return resolved

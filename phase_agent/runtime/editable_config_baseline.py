"""Compare editable files with confirmed settings without changing their authority."""

from copy import deepcopy
from pathlib import Path

from phase_agent.configuration.session.project_config_json import (
    FORMAT_ID,
    profile_defaults,
)
from phase_agent.configuration.session.load_editable_config_json import (
    _DYNAMIC_OBJECT_PATHS,
)
from phase_agent.configuration.session.split_project_config import read_document
from phase_agent.configuration.session.relocate_workspace_paths import (
    relocate_workspace_paths,
)


def editable_comparison_baseline(config, source, project_root):
    """Add inherited fields and explicit local relocations for comparison only."""
    expected = deepcopy(config)
    source = Path(source)
    if source.is_file() and read_document(source).get("_format") == FORMAT_ID:
        expected = _add_inherited_fields(expected, profile_defaults())
    root = Path(project_root).resolve()
    settings_path = root / "agent_runtime.json"
    if settings_path.is_file():
        settings = read_document(settings_path)
        if settings.get("local_path_relocations"):
            expected = relocate_workspace_paths(
                expected, settings["local_path_relocations"], root
            )
        if settings.get("runtime_storage_override") is not None:
            expected["storage"] = deepcopy(settings["runtime_storage_override"])
            # A selected diagram has its own runtime root. The editable project
            # settings still describe the parent workspace, not that run namespace.
            expected["storage"]["workspace_root"] = str(root)
    return expected


def _add_inherited_fields(confirmed, defaults, path="config"):
    # Element/phase/ledger dictionaries are complete user selections, not schemas.
    atomic = _DYNAMIC_OBJECT_PATHS | {
        "config.system.boundary",
        "config.system.constraints.TM_ratio",
        "config.round_strategy.rule_default.generation_quotas",
    }
    if path in atomic:
        return deepcopy(confirmed)
    result = deepcopy(confirmed)
    for key, value in defaults.items():
        if key not in result:
            result[key] = deepcopy(value)
        elif isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _add_inherited_fields(result[key], value, f"{path}.{key}")
    return result

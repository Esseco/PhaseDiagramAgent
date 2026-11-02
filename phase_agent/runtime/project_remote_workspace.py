"""Per-project deployment location; independent of immutable science snapshots."""

from pathlib import Path, PurePosixPath
from phase_agent.tools.step_runner.file_protocol import read_json, write_json


def remote_workspace(runtime_config):
    return read_json(Path(runtime_config), {}).get("remote_workspace")


def set_remote_workspace(runtime_config, remote_root):
    value = str(remote_root or "").strip()
    path = PurePosixPath(value)
    if (
        not value.startswith("/")
        or value.startswith("//")
        or "\\" in value
        or ".." in path.parts
        or str(path) == "/"
    ):
        raise ValueError("remote_workspace must be a project-specific absolute Linux directory")
    runtime_config = Path(runtime_config).resolve()
    if not runtime_config.is_file():
        raise FileNotFoundError(runtime_config)
    config = read_json(runtime_config, {})
    previous = config.get("remote_workspace")
    config["remote_workspace"] = str(path)
    write_json(runtime_config, config)
    return {
        "status": "remote_workspace_updated",
        "runtime_config": str(runtime_config),
        "previous": previous,
        "remote_workspace": str(path),
        "remote_checked": False,
        "uploaded": False,
        "submitted": False,
    }

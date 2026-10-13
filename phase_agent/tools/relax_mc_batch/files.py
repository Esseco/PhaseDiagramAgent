"""Portable snapshot paths and atomic JSON; transfer integrity stays in the existing helper."""

from copy import deepcopy
import json
from pathlib import Path
import shlex


PATH_KEYS = {
    "source_path",
    "structure_path",
    "full_na_structure_path",
    "full_na_structure",
    "result_path",
    "checkpoint",
    "input_path",
    "manifest_path",
    "script_path",
    "upload_directory",
    "results_directory",
    "calculation_directory",
    "path",
    "archive_csv_path",
    "current_csv_path",
    "csv_path",
    "upload_archive",
}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False, default=str
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rebase(value, root, *, portable=False):
    root = Path(root).resolve()

    def convert(item, key=None):
        if isinstance(item, dict):
            return {k: convert(v, k) for k, v in item.items()}
        if isinstance(item, list):
            return [convert(v, key) for v in item]
        if isinstance(item, str) and key in PATH_KEYS and item:
            source = Path(item)
            if portable and source.is_absolute():
                try:
                    return source.relative_to(root).as_posix()
                except ValueError:
                    return item
            if not portable and not source.is_absolute():
                resolved = (root / source).resolve()
                if not resolved.is_relative_to(root):
                    raise ValueError("snapshot_path_outside_package")
                return str(resolved)
        return deepcopy(item)

    return convert(value)


def script(profile, command, *, name):
    options = {"job-name": name, **profile.get("slurm_options", {})}
    options.pop("array", None)
    for key, value in options.items():
        if "\n" in str(key) + str(value) or "\r" in str(key) + str(value):
            raise ValueError("invalid_Slurm_directive")
    directives = "\n".join(
        f"#SBATCH --{k}" + ("" if v is True else f"={v}")
        for k, v in options.items()
        if v is not None and v is not False
    )
    preamble = "\n".join(profile.get("shell_preamble") or [])
    return f"#!/usr/bin/env bash\n{directives}\nset -euo pipefail\n{preamble}\n{command}\n"


def gpu_profile(profiles, stage="relax_and_feature"):
    return next(row for row in profiles.values() if stage in row.get("stages", []))


def python_environment(root):
    root = shlex.quote(str(Path(root).resolve()))
    return f'export PYTHONPATH={root}/runtime/mc_backend:{root}/runtime:"${{PYTHONPATH:-}}"\n'

"""Map a worker's absolute or relative output path into one task directory."""

from pathlib import PurePosixPath


def resolve_final_structure(value, task_directory):
    path = PurePosixPath(str(value).replace("\\", "/"))
    root = PurePosixPath(str(task_directory).replace("\\", "/"))
    if path.is_absolute():
        try:
            relative = path.relative_to(root)
        except ValueError:
            return None
    else:
        relative = path
    if (
        not relative.parts
        or ".." in relative.parts
        or relative.suffix.lower() not in {".vasp", ".poscar"}
    ):
        return None
    return relative

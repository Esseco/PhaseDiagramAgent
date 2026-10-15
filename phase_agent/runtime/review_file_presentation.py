"""Local review files are locations to copy, not browser hyperlinks."""

from pathlib import Path


def file_location_lines(label, path):
    path = Path(path)
    return [
        f"{label}详情已保存，用 VS Code 或记事本打开：",
        f"文件夹：`{path.parent}`",
        f"文件名：`{path.name}`",
    ]

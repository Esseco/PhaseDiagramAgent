"""Keep complete human review addresses visible in brief interaction replies."""

from pathlib import Path


def file_location_lines(label, path):
    return [f"{label}：`{Path(path).absolute()}`"]


def inventory_location_lines(text):
    return [
        line
        for line in text.splitlines()
        if line.startswith(("Branch 分布与构成文件：", "DFT 选点文件："))
    ]

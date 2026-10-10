"""Apply explicit local relocations to a copy, preserving scientific snapshots."""

from copy import deepcopy
from pathlib import Path


def relocate_workspace_paths(config, relocations, workspace):
    root = Path(workspace).resolve()
    pairs = []
    root_moves = [
        (Path(row["from"]).resolve(), Path(row["to"]).resolve())
        for row in relocations or []
        if Path(row["to"]).resolve() == root
    ]
    for row in relocations or []:
        old, new = Path(row["from"]), Path(row["to"])
        if not old.is_absolute() or not new.is_absolute():
            raise ValueError("工作区迁移映射必须为绝对路径")
        explicit_move = any(
            old.resolve().is_relative_to(source) and new.resolve().is_relative_to(target)
            for source, target in root_moves
        )
        if (
            not old.resolve().is_relative_to(root) or not new.resolve().is_relative_to(root)
        ) and not explicit_move:
            raise ValueError("工作区迁移映射不能跨工作区")
        pairs.extend(((str(old), str(new)), (old.as_posix(), new.as_posix())))

    pairs.sort(key=lambda pair: len(pair[0]), reverse=True)

    def rewrite(value):
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, str):
            for old, new in pairs:
                if value.lower() == old.lower():
                    return new
                if value.lower().startswith(
                    old.lower().rstrip("/\\") + ("/" if "/" in value else "\\")
                ):
                    return new.rstrip("/\\") + value[len(old.rstrip("/\\")) :]
        return deepcopy(value)

    return rewrite(config)

"""Apply explicit local relocations to a copy, preserving scientific snapshots."""

from copy import deepcopy
from pathlib import Path


def relocate_workspace_paths(config, relocations, workspace, *, preserve_snapshots=False):
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
        if old.drive.endswith(":") and not str(old).startswith("\\\\?\\"):
            extended = "\\\\?\\" + str(old)
            pairs.extend(((extended, str(new)), (extended.replace("\\", "/"), new.as_posix())))

    destinations = {old.replace("\\", "/").rstrip("/").lower(): new for old, new in pairs}

    path_starts = {old[:3].lower() for old, _ in pairs}

    def rewrite(value):
        if isinstance(value, dict):
            return {
                key: deepcopy(item)
                if preserve_snapshots
                and key
                in {
                    "confirmed_config",
                    "confirmed_snapshot",
                    "source_config_snapshot",
                    "agent_proposal",
                    "raw_action",
                    "final_action",
                    "action_records",
                    "invocations",
                    "local_path_relocations",
                    "path_relocations",
                }
                else rewrite(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, str):
            if value[:3].lower() not in path_starts:
                return value
            seen = set()
            while value not in seen:
                if len(seen) > len(destinations):
                    break
                seen.add(value)
                prefix = value.replace("\\", "/").rstrip("/").lower()
                while prefix not in destinations:
                    if "/" not in prefix:
                        return value
                    prefix = prefix.rsplit("/", 1)[0]
                new = destinations[prefix]
                if "/" in value:
                    new = new.replace("\\", "/")
                else:
                    new = new.replace("/", "\\")
                candidate = new.rstrip("/\\") + value[len(prefix) :]
                if candidate == value:
                    return value
                value = candidate
            raise ValueError("工作区路径迁移映射存在循环")
        return deepcopy(value)

    return rewrite(config)

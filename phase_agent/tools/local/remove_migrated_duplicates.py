"""Delete only old backup copies with a byte-identical live counterpart."""

from collections import defaultdict
from pathlib import Path
from phase_agent.tools.local.migrate_workspace_layout import _file_hash, require_agent_offline
from phase_agent.tools.remote.migrate_legacy_upload_layout import _atomic_write


def remove_migrated_duplicates(workspace, *, apply=False, agent_port=8765):
    root = Path(workspace).resolve()
    archive = root / "history_backups"
    if not archive.is_dir():
        raise ValueError("请先完成明确目录名迁移")
    by_size = defaultdict(list)
    for directory in (
        "parameters",
        "workflow_state",
        "structures",
        "submissions",
        "analysis_outputs",
        "documentation",
    ):
        for path in (root / directory).rglob("*"):
            if path.is_symlink():
                raise ValueError(f"不处理链接：{path}")
            if path.is_file():
                by_size[path.stat().st_size].append(path)
    candidates = [p for p in archive.rglob("*") if p.is_file() and p.stat().st_size in by_size]
    hash_map = {}
    needed_sizes = {p.stat().st_size for p in candidates}
    for size in needed_sizes:
        for path in by_size[size]:
            hash_map.setdefault(_file_hash(path), path)
    rows = []
    for path in candidates:
        if path.is_symlink() or not path.resolve().is_relative_to(archive.resolve()):
            raise ValueError(f"备份路径不安全：{path}")
        digest = _file_hash(path)
        retained = hash_map.get(digest)
        if retained and path.name == retained.name:
            rows.append({"removed": str(path), "retained": str(retained), "sha256": digest})
    result = {"status": "planned", "count": len(rows), "files": rows}
    if not apply:
        return result
    require_agent_offline(agent_port)
    # Recheck all candidates before any deletion; a retained copy must still exist.
    for row in rows:
        if any(_file_hash(Path(row[key])) != row["sha256"] for key in ("removed", "retained")):
            raise RuntimeError("文件已变化，未删除旧副本")
    from datetime import datetime

    manifest = archive / (
        "deleted_duplicates_" + datetime.now().strftime("%Y%m%dT%H%M%S%f") + ".json"
    )
    _atomic_write(manifest, result)
    for row in rows:
        Path(row["removed"]).unlink()
    for path in sorted(archive.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and not path.is_symlink() and not any(path.iterdir()):
            path.rmdir()
    result.update(status="deleted", manifest=str(manifest))
    _atomic_write(manifest, result)
    return result

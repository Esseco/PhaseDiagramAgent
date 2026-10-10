"""Move legacy stage folders into allocation scopes with reversible path repair."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _relink(value, mapping):
    if isinstance(mapping, dict):
        mapping = sorted(mapping.items(), key=lambda pair: -len(pair[0]))
    if isinstance(value, str):
        for old, new in mapping:
            if value == old or value.startswith(old + "\\") or value.startswith(old + "/"):
                return new + value[len(old) :]
        return value
    if isinstance(value, list):
        return [_relink(item, mapping) for item in value]
    if isinstance(value, dict):
        return {_relink(key, mapping): _relink(item, mapping) for key, item in value.items()}
    return value


def migrate(state_path, round_directory, *, apply=False, search_groups=False):
    state_path = Path(state_path).resolve()
    round_directory = Path(round_directory).resolve()
    project_root = state_path.parent.parent
    upload_root = (project_root / "upload_batches").resolve()
    if round_directory.parent != upload_root or not round_directory.name.startswith("MLIP-round-"):
        raise ValueError("迁移对象必须是本项目 upload_batches 下明确的 MLIP 目录")
    original_state = state_path.read_bytes()
    state = json.loads(original_state)
    moves, mapping, scopes = [], {}, {}
    for group in ("Relax-screening", "MC-search"):
        stage = round_directory / group
        batches = [
            row
            for row in state.get("slurm_batches") or []
            if row.get("upload_directory")
            and Path(row["upload_directory"]).resolve().parent == stage
        ]
        if not batches:
            continue
        model = batches[0]["model_version"]
        operation_id = hashlib.sha256(f"legacy-first:{model}:{group}".encode()).hexdigest()[:12]
        label = (
            "allocation-0001" + ("_segment-01" if group == "MC-search" else "") + "_" + operation_id
        )
        destination = stage / label
        if destination.exists():
            raise FileExistsError(f"迁移目标已存在：{destination}")
        names = {Path(row["upload_directory"]).name for row in batches}
        children = list(stage.iterdir())
        legacy = [
            path for path in children if not path.name.startswith(("allocation-", "DFT-round-"))
        ]
        for source in legacy:
            if (
                source.name != "results"
                and source.name not in names
                and not (source.suffix == ".zip" and source.stem in names)
            ):
                raise ValueError(f"阶段目录包含无法确定归属的文件：{source}")
            if source.is_symlink() or any(path.is_symlink() for path in source.rglob("*")):
                raise ValueError(f"不能迁移含链接的目录：{source}")
            target = destination / source.name
            if (
                upload_root not in source.resolve().parents
                or upload_root not in target.resolve().parents
            ):
                raise ValueError("源或目标不在已验证的上传目录内")
            moves.append((source, target))
            mapping[str(source)] = str(target)
            mapping[source.as_posix()] = target.as_posix()
        scopes[group] = {
            "operation_id": operation_id,
            "operation_index": 1,
            "operation_directory": str(destination),
            "model": model,
            "batch_ids": [row["batch_id"] for row in batches],
        }
    if search_groups:
        if len(state.get("generation_history") or []) != 1:
            raise ValueError("当前迁移仅适用于已确认只有一组 branch 生成记录的项目")
        moves, mapping, scopes = [], {}, {}
        grouped = {}
        for batch in state.get("slurm_batches") or []:
            if not batch.get("upload_directory"):
                continue
            source = Path(batch["upload_directory"]).resolve().parent
            if round_directory not in source.parents or not (
                source.parent.name in {"MC-search", "Relax-screening"}
                or source.parent.name == "Search-group-0001"
            ):
                continue
            if batch.get("calculation_group") not in {"MC-search", "Relax-screening"}:
                continue
            grouped.setdefault(source, []).append(batch)
        relax_indices = {
            int(row.get("operation_index") or 1)
            for batches in grouped.values()
            for row in batches
            if row.get("calculation_group") == "Relax-screening"
        }
        if len(relax_indices) > 1:
            raise ValueError("存在多个 Relax 分配，需要逐 branch 确定父 Relax 后再迁移")
        parent_relax = next(iter(relax_indices), 1)
        for source, batches in grouped.items():
            group = batches[0]["calculation_group"]
            operation_id = batches[0]["operation_id"]
            index = int(batches[0]["operation_index"])
            if group == "MC-search":
                task_ids = {tid for batch in batches for tid in batch.get("task_ids") or []}
                segments = {
                    int(task.get("segment_index") or 0)
                    for task in state.get("tasks") or []
                    if task.get("task_id") in task_ids
                }
                if len(segments) != 1:
                    raise ValueError(f"无法确定唯一 MC 轮次：{source}")
                index = next(iter(segments)) + 1
                label = f"Relax-{parent_relax:04d}_MC-round-{index:04d}"
            else:
                label = f"Relax-{index:04d}"
            target = round_directory / "Search-group-0001" / label
            if target == source:
                continue
            if target.exists() or upload_root not in target.resolve().parents:
                raise ValueError(f"目标已存在或超出范围：{target}")
            if source.is_symlink() or any(path.is_symlink() for path in source.rglob("*")):
                raise ValueError(f"不能迁移含链接的目录：{source}")
            moves.append((source, target))
            mapping[str(source)] = str(target)
            mapping[source.as_posix()] = target.as_posix()
            scopes[str(source)] = {
                "group": group,
                "search_group_index": 1,
                "operation_id": operation_id,
                "operation_index": int(batches[0]["operation_index"]),
                "operation_directory": str(target),
                "model": batches[0]["model_version"],
                "batch_ids": [batch["batch_id"] for batch in batches],
            }
            if group == "MC-search":
                scopes[str(source)]["parent_relax_round"] = parent_relax
                scopes[str(source)]["parent_relax_directory"] = str(
                    target.parent / f"Relax-{parent_relax:04d}"
                )
    if not moves:
        return {"status": "already_organized", "moves": []}
    documents = [state_path]
    for name in ("phase_data.json", "branch_energy_pools.json", "phase_identification_cache.json"):
        path = state_path.parent / name
        if path.is_file():
            documents.append(path)
    phase_root = state_path.parent / "phase_diagrams"
    documents += sorted(
        path
        for path in phase_root.rglob("*")
        if path.is_file() and path.suffix in {".json", ".csv"}
    )
    updates = {}
    originals = {path: path.read_bytes() for path in documents}
    for path, raw in originals.items():
        if path.suffix == ".json":
            value = _relink(json.loads(raw), mapping)
            if path == state_path:
                for group, scope in scopes.items():
                    scope_group = scope.get("group", group)
                    registry_key = (
                        f"{scope['model']}:Search-group-0001:{scope_group}"
                        if search_groups
                        else f"{scope['model']}:{scope_group}"
                    )
                    registry = (
                        value.setdefault("upload_layout", {})
                        .setdefault("operations", {})
                        .setdefault(registry_key, {})
                    )
                    registry[scope["operation_id"]] = scope["operation_index"]
                    for batch in value.get("slurm_batches") or []:
                        if batch.get("batch_id") in scope["batch_ids"]:
                            batch.update(
                                {
                                    key: scope[key]
                                    for key in (
                                        "operation_id",
                                        "operation_index",
                                        "operation_directory",
                                    )
                                }
                            )
                            if search_groups:
                                batch["search_group_index"] = 1
                                if scope_group == "MC-search":
                                    batch["parent_relax_round"] = scope["parent_relax_round"]
                                    batch["parent_relax_directory"] = scope[
                                        "parent_relax_directory"
                                    ]
                    for collection in ("tasks", "pending_tasks"):
                        for task in value.get(collection) or []:
                            if task.get("slurm_batch_id") in scope["batch_ids"]:
                                task["upload_operation_id"] = scope["operation_id"]
                                if search_groups:
                                    task["search_group_index"] = 1
                                    if scope_group == "MC-search":
                                        task["parent_relax_round"] = scope["parent_relax_round"]
            if value != json.loads(raw):
                updates[path] = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode(
                    "utf-8"
                )
        else:
            text = raw.decode("utf-8-sig")
            for old, new in sorted(mapping.items(), key=lambda pair: -len(pair[0])):
                text = text.replace(old, new)
            encoded = (
                text.encode("utf-8-sig")
                if raw.startswith(b"\xef\xbb\xbf")
                else text.encode("utf-8")
            )
            if encoded != raw:
                updates[path] = encoded
    inventory = {
        str(source): {
            str(path.relative_to(source)): _digest(path)
            for path in source.rglob("*")
            if path.is_file()
        }
        if source.is_dir()
        else {"": _digest(source)}
        for source, _ in moves
    }
    summary = {
        "moves": [{"source": str(src), "target": str(dst)} for src, dst in moves],
        "updated_documents": [str(path) for path in updates],
        "batch_count": sum(len(row["batch_ids"]) for row in scopes.values()),
        "file_count": sum(len(files) for files in inventory.values()),
    }
    if not apply:
        return {"status": "validated_only", **summary}
    if any(path.read_bytes() != raw for path, raw in originals.items()):
        raise RuntimeError("项目文档在预检期间变化，停止迁移")
    backup = (
        project_root
        / "backups"
        / "upload_layout"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    backup.mkdir(parents=True, exist_ok=False)
    for path in updates:
        saved = backup / path.relative_to(project_root)
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, saved)
    (backup / "migration.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    moved, written = [], []
    try:
        for source, target in moves:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
            moved.append((source, target))
            actual = (
                {
                    str(path.relative_to(target)): _digest(path)
                    for path in target.rglob("*")
                    if path.is_file()
                }
                if target.is_dir()
                else {"": _digest(target)}
            )
            if actual != inventory[str(source)]:
                raise RuntimeError(f"迁移后文件校验失败：{target}")
        for path in [p for p in updates if p != state_path] + [state_path]:
            if path not in updates:
                continue
            if path.read_bytes() != originals[path]:
                raise RuntimeError(f"项目文档并发变化，停止迁移：{path}")
            temporary = path.with_name(path.name + ".allocation-layout.tmp")
            temporary.write_bytes(updates[path])
            temporary.replace(path)
            written.append(path)
    except Exception:
        for path in reversed(written):
            if path.read_bytes() != updates[path]:
                raise RuntimeError(f"回滚时检测到并发修改；请用备份恢复：{backup}")
            shutil.copy2(backup / path.relative_to(project_root), path)
        for source, target in reversed(moved):
            shutil.move(str(target), str(source))
        for directory in {target.parent for _, target in moves}:
            if directory.is_dir() and not list(directory.iterdir()):
                directory.rmdir()
        raise
    return {"status": "migrated", "backup_directory": str(backup), **summary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--round-directory", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--search-groups", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            migrate(
                args.state, args.round_directory, apply=args.apply, search_groups=args.search_groups
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Copy validated MLIP results into one shared results/ per stage folder.

The original task folders and immutable upload manifests are deliberately kept.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from execution_layer.remote.integrity import file_checksum, verify_result


STAGES = {"relax_and_feature", "deep_search"}
OPTIONAL_NAMES = (
    "status.json.gz", "settings.json.gz", "trace.json.gz",
    "task.stdout.log", "task.stderr.log", "checkpoint.json",
    "checkpoint.json.gz",
)


def plan_migration(state_path, round_directory):
    state_path = Path(state_path).resolve()
    round_directory = Path(round_directory).resolve()
    original = state_path.read_bytes()
    state = json.loads(original)
    tasks = []
    for task in state.get("tasks") or []:
        if task.get("stage") not in STAGES or not task.get("input_path"):
            continue
        task_dir = Path(task["input_path"]).resolve().parent
        if round_directory not in task_dir.parents:
            continue
        batch_dir = task_dir.parent
        if not (batch_dir / "manifest.json").is_file():
            raise ValueError(f"批次清单缺失：{batch_dir}")
        snapshot = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
        source_result = task_dir / "result.json"
        source_marker = task_dir / "task.finished.json"
        expected = {key: snapshot.get(key) for key in (
            "task_id", "task_key", "batch_id", "config_version", "model_version",
            "protocol_version", "input_file_version", "task_checksum")}
        checked = verify_result(source_result, source_marker, expected)
        if not checked["valid"]:
            raise ValueError(f"结果校验失败 {task_dir}: {checked['reason']}")
        result = checked["result"]
        files = [source_result, source_marker]
        if result.get("status") == "completed":
            outputs = result.get("outputs") or {}
            raw = outputs.get("structure_path") or outputs.get("final_structure_path")
            if not raw:
                raise ValueError(f"完成任务缺最终结构路径：{task_dir}")
            name = Path(str(raw).replace("\\", "/")).name
            candidates = (task_dir / name, task_dir / "pool" / name)
            structure = next((path for path in candidates if path.is_file()), None)
            if structure is None:
                raise FileNotFoundError(f"最终结构缺失：{task_dir / name}")
            checksum = outputs.get("structure_checksum")
            if checksum and file_checksum(structure) != checksum:
                raise ValueError(f"最终结构校验失败：{structure}")
            files.append(structure)
        files.extend(task_dir / name for name in OPTIONAL_NAMES
                     if (task_dir / name).is_file())
        pool_summary = task_dir / "pool" / "pool_summary.json.gz"
        if pool_summary.is_file():
            files.append(pool_summary)
        destination = batch_dir.parent / "results" / task_dir.name
        names = [path.name for path in files]
        if len(names) != len(set(names)):
            raise ValueError(f"结果文件名冲突：{task_dir}")
        for source in files:
            target = destination / source.name
            if target.is_file() and file_checksum(target) != file_checksum(source):
                raise ValueError(f"目标已有不同内容，停止迁移：{target}")
        tasks.append((task, batch_dir, destination, files))
    if not tasks:
        raise ValueError(f"未找到本轮现有任务：{round_directory}")
    _validate_redundant_batch_results(tasks)
    return original, state, tasks


def migrate(state_path, round_directory, *, apply=False):
    state_path = Path(state_path).resolve()
    original, state, tasks = plan_migration(state_path, round_directory)
    batches = {str(batch) for _, batch, _, _ in tasks}
    summary = {"tasks": len(tasks), "batches": len(batches),
               "pending": sum(task.get("status") == "pending" for task, _, _, _ in tasks)}
    if not apply:
        return {"status": "validated_only", **summary}
    if state_path.read_bytes() != original:
        raise RuntimeError("state 在预检期间变化，停止迁移")
    for _, _, destination, files in tasks:
        destination.mkdir(parents=True, exist_ok=True)
        # A marker in results/ only appears after all other files are complete.
        marker = destination / "task.finished.json"
        marker.unlink(missing_ok=True)
        for source in files:
            if source.name != marker.name:
                _copy_if_needed(source, destination / source.name)
        _copy_if_needed(next(path for path in files if path.name == marker.name), marker)
    updated = deepcopy(state)
    paths = {task["task_id"]: destination / "result.json"
             for task, _, destination, _ in tasks}
    for collection in (updated.get("tasks") or [], updated.get("pending_tasks") or []):
        for task in collection:
            new_path = paths.get(task.get("task_id"))
            if new_path is None:
                continue
            if task.get("status") in {"pending", "running"}:
                task["result_path"] = str(new_path)
            elif task.get("calculation_result_path"):
                task["calculation_result_path"] = str(new_path)
    for batch in updated.get("slurm_batches") or []:
        directory = batch.get("upload_directory")
        if directory and str(Path(directory).resolve()) in batches:
            batch["results_directory"] = str(Path(directory).parent / "results")
    if state_path.read_bytes() != original:
        raise RuntimeError("state 在复制期间变化，结果已保留但未改 state")
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = state_path.with_name(f"state.before-results-layout-{suffix}.json")
    if backup.exists():
        raise FileExistsError(backup)
    shutil.copy2(state_path, backup)
    temporary = state_path.with_name(state_path.name + ".results-layout.tmp")
    temporary.write_text(json.dumps(updated, ensure_ascii=False, indent=2, default=str) + "\n",
                         encoding="utf-8")
    temporary.replace(state_path)
    removed = []
    for batch in {batch_dir for _, batch_dir, _, _ in tasks}:
        legacy = batch / "results"
        if legacy.is_dir():
            shutil.rmtree(legacy)
            removed.append(str(legacy))
    return {"status": "migrated", **summary, "state_backup": str(backup),
            "removed_duplicate_batch_results": removed}


def _copy_if_needed(source, destination):
    if destination.is_file():
        return
    temporary = destination.with_name(destination.name + ".part")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def _validate_redundant_batch_results(tasks):
    by_batch = {}
    for _, batch, destination, files in tasks:
        by_batch.setdefault(batch, {})[destination.name] = {
            source.name: file_checksum(source) for source in files
        }
    for batch, expected_tasks in by_batch.items():
        legacy = batch / "results"
        if not legacy.is_dir():
            continue
        top_level_files = {path.name for path in legacy.iterdir() if path.is_file()}
        if top_level_files:
            raise ValueError(f"旧批次 results 含有未识别文件：{legacy}")
        actual_tasks = {path.name for path in legacy.iterdir() if path.is_dir()}
        if actual_tasks != set(expected_tasks):
            raise ValueError(f"旧批次 results 含有迁移范围外目录：{legacy}")
        for task_name, expected_files in expected_tasks.items():
            folder = legacy / task_name
            actual_files = {path.name for path in folder.rglob("*") if path.is_file()}
            if actual_files != set(expected_files):
                raise ValueError(f"旧批次 results 含有迁移范围外文件：{folder}")
            for name, checksum in expected_files.items():
                if file_checksum(folder / name) != checksum:
                    raise ValueError(f"旧批次结果与本地原件不同，不能清理：{folder / name}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--round-directory", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(migrate(args.state, args.round_directory, apply=args.apply),
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

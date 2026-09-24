"""Execute one manifest entry and atomically publish its normalized result."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import traceback
from typing import Any, Callable


TERMINAL_STATUSES = {"completed", "failed", "timeout", "cancelled"}


def run_slurm_array_task(
    manifest_path: str | Path,
    array_index: int,
    *,
    executor: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    """Run exactly one task; failures are also materialized for recovery."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    try:
        entry = next(row for row in manifest if int(row["array_index"]) == int(array_index))
    except StopIteration as error:
        raise IndexError(f"manifest 中不存在 array index {array_index}") from error
    input_path = Path(entry["input_path"])
    result_path = Path(entry["result_path"])
    task = json.loads(input_path.read_text(encoding="utf-8"))
    try:
        result = executor(task)
        if not isinstance(result, dict):
            raise TypeError("worker executor 必须返回 dict")
        status = result.get("status")
        if status not in TERMINAL_STATUSES:
            raise ValueError(f"worker 必须写终态结果，当前 status={status!r}")
        payload = {**result, "task_id": entry["task_id"], "task_key": entry["task_key"]}
    except Exception as error:  # cluster workers must always leave a recoverable result
        payload = {
            "status": "failed",
            "task_id": entry["task_id"],
            "task_key": entry["task_key"],
            "actual_cost": None,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
    _write_json(result_path, payload)
    _write_json(result_path.with_name("task.finished.json"), {
        "task_id": entry["task_id"], "task_key": entry["task_key"],
        "status": payload["status"], "result_file": result_path.name,
    })
    return payload


def load_executor(reference: str):
    """Load ``package.module:function`` without coupling the runner to a backend."""
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("executor 必须使用 package.module:function 格式")
    executor = getattr(importlib.import_module(module_name), attribute)
    if not callable(executor):
        raise TypeError(f"executor 不可调用：{reference}")
    return executor


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run one Slurm array manifest task")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--array-index", required=True, type=int)
    parser.add_argument("--executor", required=True, help="package.module:function")
    arguments = parser.parse_args(argv)
    result = run_slurm_array_task(
        arguments.manifest, arguments.array_index,
        executor=load_executor(arguments.executor),
    )
    return 0 if result["status"] == "completed" else 1


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())

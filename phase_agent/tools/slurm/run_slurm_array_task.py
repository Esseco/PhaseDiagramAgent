"""Execute one manifest entry and atomically publish its normalized result."""

from __future__ import annotations

import argparse
from phase_agent.module_references import import_saved_module
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
    root = Path(manifest_path).resolve().parent
    input_path = Path(entry["input_path"])
    download_result_path = Path(entry["result_path"])
    if not input_path.is_absolute():
        input_path = root / input_path
    if not download_result_path.is_absolute():
        download_result_path = root / download_result_path
    consolidated = download_result_path.parent.parent.name == "results"
    result_path = input_path.parent / "result.json" if consolidated else download_result_path
    task = json.loads(input_path.read_text(encoding="utf-8"))
    task["result_path"] = str(result_path)
    if task.get("calculation_directory"):
        calculation_directory = Path(task["calculation_directory"])
        if not calculation_directory.is_absolute():
            task["calculation_directory"] = str(root / calculation_directory)
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
    for key in (
        "batch_id",
        "config_version",
        "model_version",
        "task_checksum",
        "protocol_version",
        "input_file_version",
    ):
        if key in entry:
            payload[key] = entry[key]
    _write_json(result_path, payload)
    from phase_agent.tools.remote.integrity import file_checksum

    marker = {
        key: payload.get(key)
        for key in (
            "task_id",
            "task_key",
            "batch_id",
            "config_version",
            "model_version",
            "task_checksum",
            "protocol_version",
            "input_file_version",
            "status",
        )
    }
    marker.update({"result_file": result_path.name, "result_checksum": file_checksum(result_path)})
    _write_json(result_path.with_name("task.finished.json"), marker)
    if consolidated:
        from phase_agent.tools.remote.export_batch_result import export_batch_result

        export_batch_result(input_path.parent, download_result_path.parent.parent)
    return payload


def load_executor(reference: str):
    """Load ``package.module:function`` without coupling the runner to a backend."""
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("executor 必须使用 package.module:function 格式")
    executor = getattr(import_saved_module(module_name), attribute)
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
        arguments.manifest,
        arguments.array_index,
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

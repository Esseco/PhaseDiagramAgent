"""Run one self-contained task directory without a parent batch manifest."""

from __future__ import annotations

import argparse
from phase_agent.module_references import import_saved_module
import json
from pathlib import Path
import traceback

from phase_agent.tools.remote.integrity import file_checksum
from phase_agent.tools.cost.runtime_observation import start_timer, runtime_observation


TERMINAL_STATUSES = {"completed", "failed", "timeout", "cancelled"}


def run_single_task(task_path="task.json", *, executor):
    path = Path(task_path).resolve()
    directory = path.parent
    task = json.loads(path.read_text(encoding="utf-8"))
    result_path = directory / "result.json"
    task["result_path"] = str(result_path)
    task["calculation_directory"] = str(directory)
    timer = start_timer()
    try:
        result = executor(task)
        if not isinstance(result, dict) or result.get("status") not in TERMINAL_STATUSES:
            raise ValueError("executor 必须返回带终态 status 的 dict")
        payload = {**result, "task_id": task["task_id"], "task_key": task["task_key"]}
    except Exception as error:
        payload = {
            "status": "failed",
            "task_id": task.get("task_id"),
            "task_key": task.get("task_key"),
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
        if task.get(key) is not None:
            payload[key] = task[key]
    payload["runtime_observation"] = runtime_observation(timer, task, payload)
    _write(result_path, payload)
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
    _write(directory / "task.finished.json", marker)
    if (directory.parent / "batch.snapshot.json").is_file():
        from phase_agent.tools.remote.export_batch_result import export_batch_result

        export_batch_result(directory, directory.parent.parent / "results")
    return payload


def _load(reference):
    module, separator, name = reference.partition(":")
    if not separator:
        raise ValueError("executor 必须使用 package.module:function 格式")
    value = getattr(import_saved_module(module), name)
    if not callable(value):
        raise TypeError(f"executor 不可调用：{reference}")
    return value


def _write(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="task.json")
    parser.add_argument("--executor", required=True)
    arguments = parser.parse_args(argv)
    result = run_single_task(arguments.task, executor=_load(arguments.executor))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

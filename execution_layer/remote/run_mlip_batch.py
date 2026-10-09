"""Run every task in one manually submitted MLIP job, preserving task results."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from execution_layer.remote.export_batch_result import export_batch_result
from execution_layer.cost.runtime_observation import start_timer, runtime_observation


def run_mlip_batch(manifest_path="manifest.json", *, executor=None):
    manifest = Path(manifest_path).resolve()
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    batch = manifest.parent
    results_directory = (batch.parent.parent if batch.parent.name == "inputs" else batch.parent) / "results"
    results_directory.mkdir(exist_ok=True)
    failures = []
    timer = start_timer()
    executed = 0
    for entry in entries:
        # Batch manifests may have been generated on Windows and uploaded as-is.
        # Treat both slash styles as portable separators on the Linux worker.
        input_path = str(entry["input_path"]).replace("\\", "/")
        task_dir = (manifest.parent / input_path).parent
        marker = task_dir / "task.finished.json"
        if marker.is_file() and (task_dir / "result.json").is_file():
            try:
                if json.loads(marker.read_text(encoding="utf-8")).get("status") == "completed":
                    export_batch_result(task_dir, results_directory)
                    continue
            except (OSError, ValueError) as error:
                print(f"已有结果汇集失败 {entry['task_id']}: {error}", file=sys.stderr)
                failures.append(entry["task_id"])
                continue
        with (task_dir / "task.stdout.log").open("a", encoding="utf-8") as stdout, \
                (task_dir / "task.stderr.log").open("a", encoding="utf-8") as stderr:
            command = ([sys.executable, "-m", "execution_layer.remote.run_single_task",
                        "--task", "task.json", "--executor", executor] if executor
                       else [sys.executable, "run_mlip_task.py"])
            outcome = subprocess.run(
                command, cwd=task_dir,
                stdout=stdout, stderr=stderr, check=False,
            )
            executed += 1
        if outcome.returncode != 0 or not marker.is_file():
            failures.append(entry["task_id"])
        if marker.is_file() and (task_dir / "result.json").is_file():
            try:
                export_batch_result(task_dir, results_directory)
            except (OSError, ValueError) as error:
                print(f"结果汇集失败 {entry['task_id']}: {error}", file=sys.stderr)
                if entry["task_id"] not in failures:
                    failures.append(entry["task_id"])
    timing = runtime_observation(timer, {"batch_id": entries[0].get("batch_id") if entries else None}, {})
    timing.update({"allocation_basis": "whole_batch_do_not_add_to_task_slices",
                   "executed_tasks": executed, "manifest_tasks": len(entries)})
    batch_id = manifest.parent.name
    try:
        (results_directory / f"{batch_id}.runtime.json").write_text(
            json.dumps(timing, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as error:
        print(f"批次统计未保存（计算结果保持不变）: {error}", file=sys.stderr)
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="manifest.json")
    parser.add_argument("--executor", default=None)
    args = parser.parse_args(argv)
    failures = run_mlip_batch(args.manifest, executor=args.executor)
    if failures:
        print("未完成的 task: " + ", ".join(failures), file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run every task in one manually submitted MLIP job, preserving task results."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def run_mlip_batch(manifest_path="manifest.json", *, executor=None):
    manifest = Path(manifest_path).resolve()
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    failures = []
    for entry in entries:
        # Batch manifests may have been generated on Windows and uploaded as-is.
        # Treat both slash styles as portable separators on the Linux worker.
        input_path = str(entry["input_path"]).replace("\\", "/")
        task_dir = (manifest.parent / input_path).parent
        marker = task_dir / "task.finished.json"
        if marker.is_file() and (task_dir / "result.json").is_file():
            try:
                if json.loads(marker.read_text(encoding="utf-8")).get("status") == "completed":
                    continue
            except (OSError, ValueError):
                pass
        with (task_dir / "task.stdout.log").open("a", encoding="utf-8") as stdout, \
                (task_dir / "task.stderr.log").open("a", encoding="utf-8") as stderr:
            command = ([sys.executable, "-m", "execution_layer.remote.run_single_task",
                        "--task", "task.json", "--executor", executor] if executor
                       else [sys.executable, "run_mlip_task.py"])
            outcome = subprocess.run(
                command, cwd=task_dir,
                stdout=stdout, stderr=stderr, check=False,
            )
        if outcome.returncode != 0 or not marker.is_file():
            failures.append(entry["task_id"])
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

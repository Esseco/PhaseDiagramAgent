"""Portable single-model Slurm array worker; uses the already activated environment."""

import json
import os
from pathlib import Path
import subprocess
import sys


def train(root, index):
    root = Path(root).resolve()
    plan = json.loads((root / "training_plan.json").read_text(encoding="utf-8"))
    names = plan["execution"]["training_models"]
    if type(index) is not int or not 0 <= index < len(names):
        raise ValueError("Invalid SLURM_ARRAY_TASK_ID")
    name = names[index]
    directory = (root / name).resolve()
    if directory.parent != root or not (directory / "mace_train.yaml").is_file():
        raise ValueError("Invalid training model directory")
    marker = directory / "training_task_status.json"
    record = {
        "model_id": name,
        "array_task_id": index,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "status": "running",
        "exit_code": None,
    }

    def save():
        temporary = marker.with_suffix(".tmp")
        temporary.write_text(json.dumps(record), encoding="utf-8")
        os.replace(temporary, marker)

    save()
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "mace.cli.run_train", "--config", "mace_train.yaml"],
            cwd=directory,
        )
        record.update(
            status="completed" if completed.returncode == 0 else "failed",
            exit_code=completed.returncode,
        )
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
    return completed.returncode


if __name__ == "__main__":
    sys.exit(train(Path(__file__).resolve().parent, int(os.environ["SLURM_ARRAY_TASK_ID"])))

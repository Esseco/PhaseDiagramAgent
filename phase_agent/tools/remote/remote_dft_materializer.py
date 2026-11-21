"""Lightweight local DFT manifest packaging and HPC CPU batch materialization."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from phase_agent.tools.remote.integrity import file_checksum
from phase_agent.tools.remote.transfer_bundle import create_bundle

DFT_MATERIALIZE_SCHEMA = "remote-dft-materialize-v1"


def prepare_remote_dft_manifest_bundle(
    tasks: list[dict],
    output_dir: str | Path,
    *,
    comparison_model: dict | None = None,
) -> dict:
    """Package one manifest + input .vasp frames locally; materialize 100 dirs on HPC CPU."""
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    frames_dir = out / "frames"
    frames_dir.mkdir(exist_ok=True)
    manifest_tasks = []
    files = []
    for idx, task in enumerate(tasks):
        src = Path(task["structure_path"])
        if not src.is_file():
            raise FileNotFoundError(f"dft structure missing: {src}")
        dst = frames_dir / f"{task['task_id']}.vasp"
        shutil.copy2(src, dst)
        sha = file_checksum(dst)
        expected = task.get("input_structure_sha256")
        if expected and sha != expected:
            raise ValueError(f"dft_input_structure_sha256_mismatch:{task['task_id']}")
        files.append(dst)
        manifest_tasks.append(
            {
                "task_id": task["task_id"],
                "task_key": task.get("task_key"),
                "stage": task.get("stage", "dft_single_point"),
                "structure_path": f"frames/{dst.name}",
                "input_structure_sha256": sha,
                "parameters": task.get("parameters") or {},
                "model_version": task.get("model_version"),
            }
        )
    manifest = {
        "schema": DFT_MATERIALIZE_SCHEMA,
        "prep_resource_type": "cpu",
        "compute_resource_type": "gpu",
        "comparison_model": comparison_model,
        "tasks": manifest_tasks,
    }
    mpath = out / "dft_batch_manifest.json"
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    runner = out / "materialize_dft.py"
    runner.write_text(
        "from pathlib import Path\n"
        "from phase_agent.tools.remote.remote_dft_materializer import materialize_remote_dft_batch\n"
        "if __name__ == '__main__':\n"
        "    materialize_remote_dft_batch(Path('dft_batch_manifest.json'), Path('dft_jobs'))\n",
        encoding="utf-8",
    )
    prep_sh = out / "prepare_cpu.sh"
    prep_sh.write_text(
        "#!/usr/bin/env bash\n"
        "#SBATCH --job-name=dft-prep-cpu\n"
        "#SBATCH --ntasks=1\n"
        "#SBATCH --cpus-per-task=4\n"
        "set -euo pipefail\n"
        'cd -- "${SLURM_SUBMIT_DIR:?}"\n'
        "python3 materialize_dft.py\n",
        encoding="utf-8",
        newline="\n",
    )
    files.extend([mpath, runner, prep_sh])
    archive = create_bundle(out, files, out / "inputs.zip", kind="inputs")
    return {
        "status": "prepared",
        "prep_resource_type": "cpu",
        "compute_resource_type": "gpu",
        "task_count": len(manifest_tasks),
        "upload_archive": str(archive),
        "manifest_path": str(mpath),
        "script_path": str(prep_sh),
    }


def materialize_remote_dft_batch(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    preparer_fn=None,
) -> dict:
    """Materialize DFT task directories on HPC CPU before GPU VASP submission."""
    mpath = Path(manifest_path).resolve()
    root = mpath.parent
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    if manifest.get("schema") != DFT_MATERIALIZE_SCHEMA:
        raise ValueError("remote_dft_materialize_schema_invalid")
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    materialized = []
    for task in manifest.get("tasks") or []:
        fpath = (root / task["structure_path"]).resolve()
        if not fpath.is_file() or file_checksum(fpath) != task["input_structure_sha256"]:
            raise ValueError("remote_dft_materialize_checksum_mismatch:" + str(task["task_id"]))
        work_dir = out / str(task["task_id"])
        work_dir.mkdir(parents=True, exist_ok=True)
        task_payload = {
            **task,
            "structure_path": str(fpath),
            "work_directory": str(work_dir),
        }
        if preparer_fn is not None:
            res = preparer_fn(task_payload, comparison_model=manifest.get("comparison_model"))
        else:
            from phase_agent.science.dft.prepare_pycode_relax import prepare_pycode_relax

            res = prepare_pycode_relax(
                task_payload,
                comparison_model=manifest.get("comparison_model"),
            )
        materialized.append(
            {"task_id": task["task_id"], "work_directory": str(work_dir), "result": res}
        )
    return {
        "status": "materialized",
        "prep_resource_type": "cpu",
        "compute_resource_type": "gpu",
        "task_count": len(materialized),
        "tasks": materialized,
    }

"""Portable task export for explicit, manual supercomputer upload."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from execution_layer.remote.batch_runner import RemoteBatchRunner, _compatibility
from execution_layer.remote.write_unix_shell_script import write_unix_shell_script


class ManualUploadBatchRunner(RemoteBatchRunner):
    """Group compatible MLIP tasks per job; keep DFT at one task per job."""

    def _select(self, state, tasks):
        selected = super()._select(state, tasks)
        if not selected:
            return selected
        first = selected[0]
        if first.get("stage") == "relax_and_feature":
            return selected  # Exactly up to 100 structures per job; branch is not a job boundary.
        if first.get("stage") in {"dft_single_point", "dft_relax"} or not first.get("branch_id"):
            return selected[:1] if first.get("stage") in {"dft_single_point", "dft_relax"} else selected
        # Avoid splitting a branch's initial states across two jobs when the
        # configured capacity cuts through its final group.
        last_branch = selected[-1].get("branch_id")
        if last_branch and any(task.get("branch_id") != last_branch for task in selected):
            selected_ids = {task["task_id"] for task in selected}
            reservations = state.get("budget_reservations") or {}
            remaining_same_branch = any(
                task.get("branch_id") == last_branch
                and task.get("task_id") not in selected_ids
                and task.get("status") == "pending"
                and not task.get("slurm_batch_id")
                and _compatibility(task) == _compatibility(first)
                and (reservations.get(task.get("task_key")) or {}).get("status")
                in {"reserved", "submitted"}
                for task in tasks
            )
            if remaining_same_branch:
                selected = [task for task in selected if task.get("branch_id") != last_branch]
        return selected

    def _script(self, batch_id, count, stage):
        return "# Generated transiently; manual tasks use their own GPU.sh.\n"

    def prepare(self, state):
        selected = self._select(state, state.get("tasks") or [])
        if selected and selected[0].get("stage") not in {"dft_single_point", "dft_relax"}:
            if self.task_preparer is None:
                raise ValueError("MLIP task_preparer 未配置，不能生成可运行任务")
            _executor_reference(self.worker_command)
        result = super().prepare(state)
        batch = result.get("batch")
        if result.get("status") != "prepared" or not batch:
            return result
        directory = Path(batch["manifest_path"]).parent
        manifest = json.loads(Path(batch["manifest_path"]).read_text(encoding="utf-8"))
        task_directories = []
        mlip_batch = manifest[0]["stage"] not in {"dft_single_point", "dft_relax"}
        for entry in manifest:
            task_directory = directory / Path(entry["input_path"]).parent
            task_directories.append(str(task_directory))
            stage = entry["stage"]
            template = Path(__file__).with_name(
                "vasp_gpu_template.sh" if stage in {"dft_single_point", "dft_relax"}
                else "mlip_gpu_template.sh"
            )
            if stage in {"dft_single_point", "dft_relax"}:
                if (json.loads((task_directory / "task.json").read_text(encoding="utf-8"))
                        .get("atomate", {}).get("backend") in {"pycode_atomate2_relax", "pycode_atomate2_static"}):
                    write_unix_shell_script(task_directory / "GPU.sh",
                        (task_directory / "submit_gpu.sh").read_text(encoding="utf-8"))
                    continue
                missing = [name for name in ("POSCAR", "INCAR", "KPOINTS", "POTCAR")
                           if not (task_directory / name).is_file()]
                if missing:
                    raise RuntimeError(f"atomate 未在 DFT 任务目录生成输入文件：{missing}")
            if not mlip_batch:
                job_name = _slurm_job_name(stage, entry["task_id"])
                script = _render_job_name(template.read_text(encoding="utf-8"), job_name)
                vasp_command = "mpirun -np $SLURM_NPROCS vasp_std"
                if script.count(vasp_command) != 1:
                    raise ValueError("DFT GPU 模板必须恰有一处 VASP 入口")
                script = script.replace(vasp_command,
                    'cd "${SLURM_SUBMIT_DIR:-.}"\n'
                    'export PHASE_TASK_STARTED_EPOCH=$(date +%s)\n'
                    'set +e\n' + vasp_command + '\nvasp_exit_code=$?\nset -e\n'
                    'python3 -m execution_layer.remote.finalize_vasp '
                    '--directory . --exit-code "$vasp_exit_code"')
                write_unix_shell_script(task_directory / "GPU.sh", script)
        if mlip_batch:
            executor = _executor_reference(self.worker_command)
            (directory / "run_mlip_batch.py").write_text(
                "from pathlib import Path\n"
                "from execution_layer.remote.run_mlip_batch import main\n"
                "if __name__ == '__main__':\n"
                "    manifest = str(Path(__file__).resolve().with_name('manifest.json'))\n"
                f"    raise SystemExit(main(['--manifest', manifest, '--executor', {executor!r}]))\n",
                encoding="utf-8",
            )
            template = Path(__file__).with_name("mlip_gpu_template.sh").read_text(encoding="utf-8")
            command = "python3 run_mlip_task.py"
            if template.count(command) != 1:
                raise ValueError("MLIP GPU 模板必须恰有一处单任务入口")
            job_name = _slurm_job_name(manifest[0]["stage"], batch["batch_id"])
            write_unix_shell_script(
                directory / "GPU.sh",
                _render_job_name(template.replace(command, "python3 run_mlip_batch.py"), job_name),
            )
        # The parent runner creates an array script; manual mode never exposes it.
        (directory / "submit.sbatch").unlink()
        checksums = []
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            if path.name == "SHA256SUMS":
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            checksums.append(f"{digest}  {path.relative_to(directory).as_posix()}")
        checksum_path = directory / "SHA256SUMS"
        checksum_path.write_text("\n".join(checksums) + "\n", encoding="utf-8")
        guide_path = directory / "UPLOAD_AND_SUBMIT.md"
        guide_path.write_text(_guide(batch["batch_id"]), encoding="utf-8")
        script_path = directory / "GPU.sh" if mlip_batch else Path(task_directories[0]) / "GPU.sh"
        results_directory = (directory.parent.parent if directory.parent.name == "inputs" else directory.parent) / "results"
        batch.update({"upload_directory": str(directory), "script_path": str(script_path),
                      "results_directory": str(results_directory),
                      "task_directory": task_directories[0], "task_directories": task_directories,
                      "checksums_path": str(checksum_path), "upload_guide": str(guide_path)})
        result["batch"] = batch
        result["state"]["slurm_batches"][-1].update({
            "upload_directory": str(directory), "script_path": str(script_path),
            "results_directory": str(results_directory),
            "task_directory": task_directories[0], "task_directories": task_directories,
            "checksums_path": str(checksum_path),
            "upload_guide": str(guide_path),
        })
        return result


def _executor_reference(command):
    try:
        return command[command.index("--executor") + 1]
    except (ValueError, IndexError):
        raise ValueError("MLIP worker_command 缺少 --executor 模块:函数") from None


def _slurm_job_name(stage, identifier):
    """Build a readable, task-specific Slurm name without unsafe characters."""
    prefix = {
        "relax_and_feature": "relax",
        "deep_search": "mc",
        "dft_single_point": "dft-sp",
        "dft_relax": "dft-relax",
    }.get(stage, "task")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", f"{prefix}-{identifier}").strip("-.")
    if len(value) > 64:
        suffix = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
        value = f"{value[:55]}-{suffix}"
    return value or "phase-task"


def _render_job_name(script, job_name):
    pattern = r"(?m)^(\s*#SBATCH\s+--job-name=).*$"
    matches = list(re.finditer(pattern, script))
    if len(matches) != 1:
        raise ValueError("GPU.sh 模板必须恰有一条 #SBATCH --job-name 指令")
    return re.sub(pattern, lambda match: f"{match.group(1)}{job_name}", script)


def _guide(batch_id):
    return f"""# Manual upload batch `{batch_id}`

This directory was generated locally. Nothing has been submitted.

1. Keep every submission directory for this stage as a sibling under the same
   allocation folder under the MLIP-round stage. Inspect `manifest.json`, `task.json`, and `GPU.sh`.
2. Confirm the model/input files and all site-specific Slurm settings.
3. Upload the complete allocation folder, keeping its batch directories as siblings. Relax jobs
   contain up to 100 structures; MC jobs contain up to 10 compatible simulations,
   possibly from multiple branches.
4. Optionally verify its files with `sha256sum -c SHA256SUMS`.
5. For MLIP Relax/MC, submit the batch root's `GPU.sh` once; it runs each task
   subdirectory and collects results in this allocation folder's shared `results/`.
   For DFT, change into the only task subdirectory and submit its `GPU.sh` once.
   Never submit both levels.
6. After all jobs for this allocation finish, download the allocation folder's single
   `results/` directory into the matching local allocation folder. It includes result JSON,
   completion markers and required final structures.

The generated script is not evidence that executables, environments, paths,
pseudopotentials, permissions, or resource requests are correct for your site.
For DFT, the remote Python environment must import this project and pymatgen
to parse VASP output and publish `results/` after the calculation.
"""

"""Materialize budget-reserved tasks as resumable Slurm batches.

The runner owns cluster I/O only.  It does not select scientific actions or
change their reserved budgets.  DFT tasks are prepared through the existing
atomate dispatcher; no second VASP input implementation is provided here.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shlex
import subprocess
from typing import Any, Callable


DFT_STAGES = {"dft_single_point", "dft_relax"}
ACTIVE_STATUSES = {"pending", "running"}


class SlurmBatchRunner:
    """Create, optionally submit, and recover file-backed Slurm batches."""

    def __init__(
        self,
        root_directory: str | Path,
        *,
        worker_command: list[str] | tuple[str, ...],
        dispatcher: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        submit: bool = False,
        sbatch_command: list[str] | tuple[str, ...] = ("sbatch",),
        max_batch_tasks: int = 100,
        stage_batch_sizes: dict[str, int] | None = None,
        max_batch_cost: float | None = None,
        slurm_options: dict[str, Any] | None = None,
        stage_profiles: dict[str, dict[str, Any]] | None = None,
        task_preparer: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        command_runner: Callable[..., Any] = subprocess.run,
    ):
        if not worker_command:
            raise ValueError("worker_command 不能为空")
        if max_batch_tasks <= 0:
            raise ValueError("max_batch_tasks 必须为正整数")
        if max_batch_cost is not None and max_batch_cost <= 0:
            raise ValueError("max_batch_cost 必须为正数")
        self.root = Path(root_directory)
        self.worker_command = [str(item) for item in worker_command]
        self.dispatcher = dispatcher
        self.submit = bool(submit)
        self.sbatch_command = [str(item) for item in sbatch_command]
        self.max_batch_tasks = int(max_batch_tasks)
        self.stage_batch_sizes = {
            str(stage): int(size) for stage, size in (stage_batch_sizes or {}).items()
        }
        if any(size <= 0 for size in self.stage_batch_sizes.values()):
            raise ValueError("stage_batch_sizes values must be positive")
        self.max_batch_cost = float(max_batch_cost) if max_batch_cost is not None else None
        self.slurm_options = dict(slurm_options or {})
        self.stage_profiles = deepcopy(stage_profiles or {})
        self.task_preparer = task_preparer
        self.command_runner = command_runner

    def prepare(self, state: dict[str, Any]) -> dict[str, Any]:
        """Prepare one deterministic batch from valid reserved pending tasks."""
        current = deepcopy(state)
        tasks = current.setdefault("tasks", deepcopy(current.get("pending_tasks") or []))
        selected = self._select_tasks(current, tasks)
        if not selected:
            return {"status": "no_tasks", "state": current, "batch": None}

        next_index = len(current.get("slurm_batches") or []) + 1
        batch_id = f"slurm-{next_index:06d}"
        batch_directory = self.root / batch_id
        batch_directory.mkdir(parents=True, exist_ok=False)
        manifest = []
        for array_index, task in enumerate(selected):
            prepared = self._prepare_task(task, batch_directory, array_index)
            manifest.append(prepared)
            stored = next(row for row in tasks if row.get("task_id") == task["task_id"])
            stored.update({
                "status": "pending",
                "slurm_batch_id": batch_id,
                "slurm_array_index": array_index,
                "input_path": prepared["input_path"],
                "result_path": prepared["result_path"],
            })

        manifest_path = batch_directory / "manifest.json"
        _write_json(manifest_path, manifest)
        is_dft_batch = selected[0].get("stage") in DFT_STAGES
        script_path = batch_directory / "submit.sbatch"
        script_path.write_text(
            self._script(batch_id, manifest_path, len(manifest), selected[0].get("stage")),
            encoding="utf-8",
        )
        script_path.chmod(0o750)
        batch = {
            "batch_id": batch_id,
            "status": "prepared",
            "task_ids": [row["task_id"] for row in manifest],
            "planned_cost": sum(row["planned_cost"] for row in manifest),
            "manifest_path": str(manifest_path),
            "backend": "atomate_vasp" if is_dft_batch else "slurm",
            "script_path": str(script_path),
            "job_id": None,
            "compatibility_key": _compatibility_key(selected[0]),
        }
        if self.submit:
            completed = self.command_runner(
                [*self.sbatch_command, str(script_path)],
                check=True, capture_output=True, text=True,
            )
            batch.update({
                "status": "submitted",
                "scheduler_job_id": _parse_job_id(completed.stdout),
                "submit_stdout": completed.stdout.strip(),
            })
            batch["job_id"] = batch["scheduler_job_id"]
        current.setdefault("slurm_batches", []).append(batch)
        current["pending_tasks"] = [row for row in tasks if row.get("status") in ACTIVE_STATUSES]
        return {"status": batch["status"], "state": current, "batch": deepcopy(batch)}

    def collect_results(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        """Read terminal result files not yet reconciled by the main loop."""
        processed = set((state or {}).get("processed_task_ids") or [])
        recovered = []
        for task in (state or {}).get("tasks") or []:
            if task.get("task_id") in processed or not task.get("result_path"):
                continue
            path = Path(task["result_path"])
            if not path.is_file():
                continue
            marker = path.with_name("task.finished.json")
            if not marker.is_file():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload.setdefault("task_id", task["task_id"])
            payload.setdefault("task_key", task["task_key"])
            if payload.get("status") in {"completed", "failed", "timeout", "cancelled"}:
                recovered.append(payload)
        return recovered

    def _select_tasks(self, state, tasks):
        reservations = state.get("budget_reservations") or {}
        selected, cost, anchor_key, batch_limit = [], 0.0, None, self.max_batch_tasks
        for task in tasks:
            if task.get("status") != "pending" or task.get("slurm_batch_id"):
                continue
            reservation = reservations.get(task.get("task_key")) or {}
            if reservation.get("status") not in {"reserved", "submitted"}:
                continue
            key = _compatibility_key(task)
            if anchor_key is None:
                anchor_key = key
                stage_limit = self.stage_batch_sizes.get(task.get("stage"), self.max_batch_tasks)
                batch_limit = 1 if task.get("stage") in DFT_STAGES else min(self.max_batch_tasks, stage_limit)
            if key != anchor_key:
                continue
            planned = float(
                reservation.get("reserved_cost", task.get("planned_relative_cost", 0.0)) or 0.0
            )
            if self.max_batch_cost is not None and cost + planned > self.max_batch_cost:
                continue
            selected.append({**task, "_slurm_planned_cost": planned})
            cost += planned
            if len(selected) >= batch_limit:
                break
        return selected

    def _prepare_task(self, task, batch_directory, array_index):
        normalized = deepcopy(task)
        planned_cost = float(normalized.pop("_slurm_planned_cost", 0.0))
        task_directory = batch_directory / f"{array_index:05d}-{task['task_id']}"
        task_directory.mkdir(parents=True, exist_ok=False)
        normalized["calculation_directory"] = str(task_directory)
        normalized["result_path"] = str(task_directory / "result.json")
        if task.get("stage") not in DFT_STAGES and self.task_preparer is not None:
            normalized = self.task_preparer(normalized)
        if task.get("stage") in DFT_STAGES:
            if self.dispatcher is None:
                raise ValueError("DFT Slurm 任务需要配置 atomate dispatcher")
            generated = self.dispatcher({**normalized, "work_directory": str(task_directory)})
            atomate = generated.get("outputs") or generated.get("result") or generated
            if atomate.get("generator") != "atomate" or generated.get("status") not in {"pending", "running"}:
                raise RuntimeError(f"DFT 任务必须由 atomate 生成：{generated}")
            normalized["atomate"] = atomate
        input_path = task_directory / "task.json"
        result_path = task_directory / "result.json"
        normalized["result_path"] = str(result_path)
        _write_json(input_path, normalized)
        return {
            "array_index": array_index,
            "task_id": task["task_id"],
            "task_key": task["task_key"],
            "stage": task.get("stage"),
            "planned_cost": planned_cost,
            "input_path": str(input_path),
            "result_path": str(result_path),
        }

    def _script(self, batch_id, manifest_path, count, stage):
        profile = self._profile_for_stage(stage)
        options = {
            "job-name": batch_id, "array": f"0-{count - 1}",
            **self.slurm_options, **(profile.get("slurm_options") or {}),
        }
        directives = "\n".join(
            f"#SBATCH --{key}" + ("" if value is True else f"={value}")
            for key, value in options.items() if value is not False and value is not None
        )
        command = " ".join(shlex.quote(item) for item in self.worker_command)
        preamble = "\n".join(profile.get("shell_preamble") or [])
        if preamble:
            preamble += "\n"
        project_root = shlex.quote(str(Path(__file__).resolve().parents[1]))
        preamble += f'export PYTHONPATH={project_root}:"${{PYTHONPATH:-}}"\n'
        if stage in DFT_STAGES:
            calculation_prefix = shlex.quote(str(Path(manifest_path).parent))
            vasp_command = profile.get("vasp_command")
            if not vasp_command:
                raise ValueError(f"DFT stage {stage} 缺少 vasp_command")
            parser_command = profile.get(
                "result_command", "python3 -m scientific_layer.dft.parse_vasp_result"
            )
            body = (
                f'batch_directory={calculation_prefix}\n'
                'task_prefix=$(printf "%s/%05d-" "$batch_directory" "$SLURM_ARRAY_TASK_ID")\n'
                'task_directory=$(compgen -G "${task_prefix}*" | head -n 1)\n'
                'test -n "$task_directory"\ncd "$task_directory"\n'
                f"set +e\n{vasp_command}\nvasp_exit_code=$?\nset -e\n"
                f'{parser_command} --directory . --result-path result.json --exit-code "$vasp_exit_code"\n'
                'exit "$vasp_exit_code"\n'
            )
        else:
            body = (
                f"{command} --manifest {shlex.quote(str(manifest_path))} "
                ' --array-index "${SLURM_ARRAY_TASK_ID}"\n'
            )
        return (
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            f"{directives}\n"
            f"{preamble}"
            f"{body}"
        )

    def _profile_for_stage(self, stage):
        for profile in self.stage_profiles.values():
            if stage in set(profile.get("stages") or []):
                return profile
        return {}


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_job_id(stdout: str) -> str | None:
    words = stdout.strip().split()
    return words[-1] if words and words[-1].isdigit() else None


def _compatibility_key(task):
    """Tasks share a job only when stage, model, parameters and resources match."""
    payload = {
        "stage": task.get("stage"),
        "model_version": task.get("model_version") or task.get("calculation_version"),
        "parameters": task.get("parameters") or {},
        "resource_profile": task.get("resource_profile"),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)

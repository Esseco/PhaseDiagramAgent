"""Remote-safe batch materialization with portable paths and immutable manifests."""
from copy import deepcopy
import json
from pathlib import Path
import shlex
import shutil
from execution_layer.remote.integrity import payload_checksum, verify_result

DFT_STAGES = {"dft_single_point", "dft_relax"}
PROTOCOL_VERSION = 1
DEFAULT_BATCH_SIZES = {"relax_and_feature": 100, "deep_search": 20, "dft_single_point": 1, "dft_relax": 1}


class RemoteBatchRunner:
    """Prepare portable snapshots locally; it never submits a scheduler job."""
    def __init__(self, root_directory, *, worker_command, dispatcher=None, stage_batch_sizes=None,
                 stage_profiles=None, task_preparer=None):
        self.root = Path(root_directory); self.worker_command = list(worker_command)
        self.dispatcher = dispatcher
        self.batch_sizes = {**DEFAULT_BATCH_SIZES, **(stage_batch_sizes or {})}
        self.stage_profiles = deepcopy(stage_profiles or {}); self.task_preparer = task_preparer

    def prepare(self, state):
        current = deepcopy(state); tasks = current.setdefault("tasks", [])
        selected = self._select(current, tasks)
        if not selected: return {"status": "no_tasks", "state": current, "batch": None}
        batch_id = f"remote-{len(current.get('slurm_batches') or []) + 1:06d}"
        directory = self.root / batch_id; directory.mkdir(parents=True, exist_ok=False)
        manifest = []
        for index, source in enumerate(selected):
            task = deepcopy(source); task.setdefault("config_version", current.get("confirmed_config_version"))
            task.setdefault("model_version", current.get("active_model_version"))
            task.update({"batch_id": batch_id, "slurm_array_index": index})
            task.setdefault("protocol_version", PROTOCOL_VERSION)
            task.setdefault("input_file_version", "task-json-v1")
            relative_dir = Path(f"{index:05d}-{task['task_id']}"); task_dir = directory / relative_dir; task_dir.mkdir()
            task.update({"calculation_directory": str(relative_dir), "result_path": str(relative_dir / "result.json")})
            if task.get("stage") not in DFT_STAGES and self.task_preparer: task = self.task_preparer(task)
            worker_job = task.get("worker_job") or {}
            if task.get("stage") in {"relax_and_feature", "deep_search"} and worker_job:
                original = Path(worker_job.get("structure_path") or "")
                if not original.is_file():
                    raise FileNotFoundError(f"Relax input structure missing: {original}")
                staged = task_dir / "initial.vasp"
                shutil.copy2(original, staged)
                worker_job["structure_path"] = staged.name
                worker_job["phase_references"] = {}
                full_na = (worker_job.get("parameters") or {}).get("full_na_structure")
                if full_na:
                    source_full_na = Path(full_na)
                    if not source_full_na.is_file():
                        raise FileNotFoundError(f"full Na structure missing: {source_full_na}")
                    staged_full_na = task_dir / "full_na_structure.vasp"
                    shutil.copy2(source_full_na, staged_full_na)
                    worker_job["parameters"]["full_na_structure"] = staged_full_na.name
                checkpoint = worker_job.get("checkpoint")
                if checkpoint:
                    source_checkpoint = Path(checkpoint)
                    if not source_checkpoint.is_file():
                        raise FileNotFoundError(f"MC checkpoint missing: {source_checkpoint}")
                    staged_checkpoint = task_dir / source_checkpoint.name
                    shutil.copy2(source_checkpoint, staged_checkpoint)
                    worker_job["checkpoint"] = staged_checkpoint.name
            if task.get("stage") in DFT_STAGES:
                if self.dispatcher is None: raise ValueError("remote DFT task requires atomate dispatcher")
                generated = self.dispatcher({**task, "work_directory": str(task_dir)})
                output = generated.get("outputs") or generated.get("result") or generated
                if output.get("generator") != "atomate": raise RuntimeError("DFT inputs must come from atomate")
                task["atomate"] = output
            task_path = task_dir / "task.json"
            checksum = payload_checksum(task)
            task["task_checksum"] = checksum
            _write(task_path, task)
            entry = {"array_index": index, "batch_id": batch_id, "task_id": task["task_id"],
                     "task_key": task["task_key"], "stage": task.get("stage"),
                     "config_version": task.get("config_version"), "model_version": task.get("model_version"),
                     "protocol_version": task.get("protocol_version"),
                     "input_file_version": task.get("input_file_version"),
                     "task_checksum": checksum, "input_path": str(task_path.relative_to(directory)),
                     "result_path": str((task_dir / "result.json").relative_to(directory))}
            manifest.append(entry)
            source.update({"slurm_batch_id": batch_id, "slurm_array_index": index, "batch_id": batch_id,
                           "config_version": entry["config_version"], "model_version": entry["model_version"],
                           "task_checksum": checksum, "input_path": str(task_path),
                           "result_path": str(task_dir / "result.json")})
        _write(directory / "manifest.json", manifest); checksum = payload_checksum(manifest)
        stage = selected[0].get("stage")
        (directory / "submit.sbatch").write_text(self._script(batch_id, len(manifest), stage), encoding="utf-8")
        snapshot = {"batch_id": batch_id, "config_version": manifest[0]["config_version"],
                    "model_version": manifest[0]["model_version"], "task_ids": [r["task_id"] for r in manifest],
                    "manifest_checksum": checksum, "protocol_version": PROTOCOL_VERSION}
        _write(directory / "batch.snapshot.json", snapshot)
        batch = {**snapshot, "checksum": checksum, "status": "prepared",
                 "manifest_path": str(directory / "manifest.json"), "script_path": str(directory / "submit.sbatch"),
                 "job_id": None}
        current.setdefault("slurm_batches", []).append(batch)
        current["pending_tasks"] = [row for row in tasks if row.get("status") in {"pending", "running"}]
        return {"status": "prepared", "state": current, "batch": deepcopy(batch)}

    def collect_results(self, state):
        processed = set(state.get("processed_task_ids") or []); recovered = []
        for task in state.get("tasks") or []:
            if task.get("task_id") in processed or not task.get("result_path"): continue
            result = Path(task["result_path"]); checked = verify_result(result, result.with_name("task.finished.json"), task)
            if checked["valid"] and checked["result"].get("status") in {"completed", "failed", "timeout", "cancelled"}:
                payload = checked["result"]
                if task.get("stage") in {"relax_and_feature", "deep_search"} and payload.get("status") == "completed":
                    from execution_layer.remote.resolve_local_relax_structure import resolve_local_relax_structure
                    payload = resolve_local_relax_structure(payload, result.parent)
                    if payload is None:
                        continue
                recovered.append(payload)
        return recovered

    def _select(self, state, tasks):
        reservations = state.get("budget_reservations") or {}; chosen = []; anchor = None
        gate = state.get("dedup_gate") or {}
        valid_after_dedup = set(gate.get("valid_structure_ids") or [])
        for task in tasks:
            if task.get("status") != "pending" or task.get("slurm_batch_id"): continue
            if gate and gate.get("status") != "ready" and task.get("stage") != "offline_check_dedup":
                continue
            if gate.get("status") == "ready" and task.get("stage") != "offline_check_dedup" and valid_after_dedup:
                if task.get("structure_id") and task.get("structure_id") not in valid_after_dedup:
                    continue
            if (reservations.get(task.get("task_key")) or {}).get("status") not in {"reserved", "submitted"}: continue
            candidate = _compatibility(task)
            if anchor is None: anchor = candidate
            if candidate != anchor: continue
            chosen.append(task); limit = 1 if task.get("stage") in DFT_STAGES else self.batch_sizes.get(task.get("stage"), 100)
            if len(chosen) >= limit: break
        return chosen

    def _script(self, batch_id, count, stage):
        profile = next((p for p in self.stage_profiles.values() if stage in set(p.get("stages") or [])), {})
        options = {"job-name": batch_id, "array": f"0-{count - 1}", **(profile.get("slurm_options") or {})}
        directives = "\n".join(f"#SBATCH --{k}={v}" for k, v in options.items())
        preamble = "\n".join(profile.get("shell_preamble") or [])
        root = 'batch_directory=$(cd "$(dirname "$0")" && pwd)\n'
        if stage in DFT_STAGES:
            command = profile.get("vasp_command")
            if not command: raise ValueError(f"DFT stage {stage} lacks vasp_command")
            body = ('task_prefix=$(printf "%s/%05d-" "$batch_directory" "$SLURM_ARRAY_TASK_ID")\n'
                    'task_directory=$(compgen -G "${task_prefix}*" | head -n 1)\ncd "$task_directory"\n'
                    f'set +e\n{command}\nexit_code=$?\nset -e\n'
                    'python3 -m scientific_layer.dft.parse_vasp_result --directory . --result-path result.json --exit-code "$exit_code"\n')
        else:
            command = " ".join(shlex.quote(str(item)) for item in self.worker_command)
            body = f'{command} --manifest "$batch_directory/manifest.json" --array-index "$SLURM_ARRAY_TASK_ID"\n'
        return f"#!/usr/bin/env bash\nset -euo pipefail\n{directives}\n{preamble}\n{root}{body}"


def _compatibility(task):
    return json.dumps({"stage": task.get("stage"), "model_version": task.get("model_version"),
                       "parameters": task.get("parameters") or {}, "resource_profile": task.get("resource_profile")},
                      sort_keys=True, default=str)


def _write(path, payload):
    temporary = Path(f"{path}.tmp"); temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
        sort_keys=True, default=str) + "\n", encoding="utf-8"); temporary.replace(path)

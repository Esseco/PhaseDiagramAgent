"""Remote-safe batch materialization with portable paths and immutable manifests."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shlex
import shutil
from execution_layer.mc_batch_policy import mc_batch_limit
from execution_layer.remote.integrity import payload_checksum, verify_result
from execution_layer.state.task_waiting import active_pending_tasks, awaiting_task_result

DFT_STAGES = {"dft_single_point", "dft_relax"}
PROTOCOL_VERSION = 1
DEFAULT_BATCH_SIZES = {"relax_and_feature": 100, "deep_search": 10, "dft_single_point": 1, "dft_relax": 1}


class RemoteBatchRunner:
    """Prepare portable snapshots locally; it never submits a scheduler job."""
    def __init__(self, root_directory, *, worker_command, dispatcher=None, stage_batch_sizes=None,
                 stage_profiles=None, task_preparer=None):
        self.root = Path(root_directory); self.worker_command = list(worker_command)
        self.dispatcher = dispatcher
        self.batch_sizes = {**DEFAULT_BATCH_SIZES, **(stage_batch_sizes or {})}
        self.batch_sizes["deep_search"] = mc_batch_limit(self.batch_sizes["deep_search"])
        self.stage_profiles = deepcopy(stage_profiles or {}); self.task_preparer = task_preparer
        self.last_collection_report = {}

    def prepare(self, state):
        current = deepcopy(state); tasks = current.setdefault("tasks", [])
        _assign_upload_operations(current, tasks)
        selected = self._select(current, tasks)
        if not selected: return {"status": "no_tasks", "state": current, "batch": None}
        batch_id = f"remote-{len(current.get('slurm_batches') or []) + 1:06d}"
        from execution_layer.remote.build_upload_batch_directory import build_upload_batch_directory
        model_version = selected[0].get("model_version") or current.get("active_model_version")
        directory, layout = build_upload_batch_directory(
            self.root, current, batch_id=batch_id, stage=selected[0].get("stage"),
            model_version=model_version,
            operation_id=selected[0].get("upload_operation_id"),
            segment_index=selected[0].get("segment_index"),
            search_group_index=selected[0].get("search_group_index"),
            parent_relax_round=selected[0].get("parent_relax_round"),
        )
        directory.parent.mkdir(parents=True, exist_ok=True)
        directory.mkdir(exist_ok=False)
        manifest = []
        for index, source in enumerate(selected):
            task = deepcopy(source); task.setdefault("config_version", current.get("confirmed_config_version"))
            task.setdefault("model_version", current.get("active_model_version"))
            task.update({"batch_id": batch_id, "slurm_array_index": index})
            task.setdefault("protocol_version", PROTOCOL_VERSION)
            task.setdefault("input_file_version", "task-json-v1")
            relative_dir = Path(f"{index:05d}-{task['task_id']}"); task_dir = directory / relative_dir; task_dir.mkdir()
            from execution_layer.remote.result_directory_name import result_directory_name
            result_folder = result_directory_name(task_dir, task.get("stage"))
            relative_result = Path("..") / "results" / result_folder / "result.json"
            task.update({"calculation_directory": str(relative_dir), "result_path": str(relative_result)})
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
                     "task_checksum": checksum, "input_path": task_path.relative_to(directory).as_posix(),
                     "result_path": relative_result.as_posix()}
            manifest.append(entry)
            source.update({"slurm_batch_id": batch_id, "slurm_array_index": index, "batch_id": batch_id,
                           "config_version": entry["config_version"], "model_version": entry["model_version"],
                           "task_checksum": checksum, "input_path": str(task_path),
                           "result_path": str((directory / relative_result).resolve())})
        results_directory = directory.parent / "results"
        results_directory.mkdir(exist_ok=True)
        _write(directory / "manifest.json", manifest); checksum = payload_checksum(manifest)
        stage = selected[0].get("stage")
        from execution_layer.remote.write_unix_shell_script import write_unix_shell_script
        write_unix_shell_script(directory / "submit.sbatch", self._script(batch_id, len(manifest), stage))
        snapshot = {"batch_id": batch_id, "config_version": manifest[0]["config_version"],
                    "model_version": manifest[0]["model_version"], "task_ids": [r["task_id"] for r in manifest],
                    "manifest_checksum": checksum, "protocol_version": PROTOCOL_VERSION,
                    **layout}
        _write(directory / "batch.snapshot.json", snapshot)
        batch = {**snapshot, "checksum": checksum, "status": "prepared",
                 "manifest_path": str(directory / "manifest.json"), "script_path": str(directory / "submit.sbatch"),
                 "upload_directory": str(directory), "results_directory": str(results_directory),
                 "job_id": None}
        current.setdefault("slurm_batches", []).append(batch)
        current["pending_tasks"] = active_pending_tasks(tasks)
        return {"status": "prepared", "state": current, "batch": deepcopy(batch)}

    def collect_results(self, state):
        return self.collect_results_with_report(state)["results"]

    def collect_results_with_report(self, state):
        """Read new batch results folders or legacy task-local result files."""
        processed = set(state.get("processed_task_ids") or [])
        directories = _task_directories_by_id(state)
        recovered = []
        report = {"status": "empty", "considered_count": 0, "recovered_count": 0,
                  "missing_result_count": 0, "missing_marker_count": 0,
                  "invalid_count": 0, "missing_structure_count": 0,
                  "details_preview": []}
        for task in state.get("tasks") or []:
            task_id = task.get("task_id")
            previously_processed = task_id in processed
            if not task_id or (previously_processed and task.get("stage") not in DFT_STAGES):
                continue
            if not previously_processed:
                report["considered_count"] += 1
            task_dir, snapshot, location_error = _locate_task_directory(task, directories)
            result = _task_result_path(task, task_dir)
            marker = result.with_name("task.finished.json") if result else None
            # Already received jobs are not newly missing returns. A re-extracted
            # DFT JSON may supplement magnetism, but still needs full verification.
            if previously_processed and (result is None or not result.is_file() or not marker.is_file()):
                continue
            if result is None or not result.is_file():
                report["missing_result_count"] += 1
                _add_collection_detail(report, task_id, "result_missing")
                continue
            if marker is None or not marker.is_file():
                report["missing_marker_count"] += 1
                _add_collection_detail(report, task_id, "completion_marker_missing")
                continue
            if location_error:
                report["invalid_count"] += 1
                _add_collection_detail(report, task_id, location_error)
                continue
            expected, identity_error = _expected_result_identity(task, snapshot)
            if identity_error:
                report["invalid_count"] += 1
                _add_collection_detail(report, task_id, identity_error)
                continue
            checked = verify_result(result, marker, expected)
            if not checked["valid"]:
                report["invalid_count"] += 1
                _add_collection_detail(report, task_id, checked["reason"])
                continue
            payload = checked["result"]
            if payload.get("status") not in {"completed", "failed", "timeout", "cancelled"}:
                continue
            if task.get("stage") in DFT_STAGES:
                outputs = payload.get("outputs") or {}
                if outputs.get("training_file"):
                    from execution_layer.remote.integrity import file_checksum
                    training = result.parent / "training.json"
                    if (Path(str(outputs["training_file"])).name != "training.json"
                            or not training.is_file()
                            or outputs.get("training_checksum") != file_checksum(training)):
                        report["invalid_count"] += 1
                        _add_collection_detail(report, task_id, "training_file_missing_or_checksum_mismatch")
                        continue
            if task.get("stage") in {"relax_and_feature", "deep_search"} and payload.get("status") == "completed":
                from execution_layer.remote.resolve_local_relax_structure import resolve_local_relax_structure
                payload = resolve_local_relax_structure(payload, result.parent)
                if payload is None:
                    report["missing_structure_count"] += 1
                    _add_collection_detail(report, task_id, "final_structure_missing_or_checksum_mismatch")
                    continue
            elif task.get("stage") in DFT_STAGES and payload.get("status") == "completed":
                outputs = payload.get("outputs") or {}
                raw_structure = outputs.get("structure_path") if outputs.get("structure") is None else None
                if raw_structure:
                    local_structure = result.parent / Path(str(raw_structure).replace("\\", "/")).name
                    if not local_structure.is_file():
                        report["missing_structure_count"] += 1
                        _add_collection_detail(report, task_id, "final_structure_missing")
                        continue
                    expected_checksum = outputs.get("structure_checksum")
                    if expected_checksum:
                        from execution_layer.remote.integrity import file_checksum
                        if file_checksum(local_structure) != expected_checksum:
                            report["missing_structure_count"] += 1
                            _add_collection_detail(report, task_id, "final_structure_checksum_mismatch")
                            continue
                    payload["outputs"] = {**outputs, "structure_path": str(local_structure.resolve())}
            compact = _compact_result_for_state(payload, result)
            if previously_processed:
                from execution_layer.state.dft_result_refresh import validated_magnetic_refresh
                refresh = validated_magnetic_refresh(task, compact)
                if refresh["status"] == "unchanged":
                    continue
                if refresh["status"] == "rejected":
                    report["invalid_count"] += 1
                    _add_collection_detail(report, task_id, refresh["reason"])
                    continue
                report["metadata_refresh_count"] = report.get("metadata_refresh_count", 0) + 1
            recovered.append(compact)
        report["recovered_count"] = len(recovered)
        if report["invalid_count"] or report["missing_structure_count"]:
            report["status"] = "needs_attention" if not recovered else "partial"
        elif report["missing_result_count"] or report["missing_marker_count"]:
            report["status"] = "waiting" if not recovered else "partial"
        elif recovered:
            report["status"] = "ready"
        self.last_collection_report = report
        return {"results": recovered, "report": report}

    def _select(self, state, tasks):
        reservations = state.get("budget_reservations") or {}; chosen = []; anchor = None
        gate = state.get("dedup_gate") or {}
        valid_after_dedup = set(gate.get("valid_structure_ids") or [])
        for task in tasks:
            if task.get("status") != "pending" or task.get("slurm_batch_id") or not awaiting_task_result(task): continue
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
                    'python3 -m execution_layer.remote.finalize_vasp --directory . --exit-code "$exit_code"\n')
        else:
            command = " ".join(shlex.quote(str(item)) for item in self.worker_command)
            body = f'{command} --manifest "$batch_directory/manifest.json" --array-index "$SLURM_ARRAY_TASK_ID"\n'
        return f"#!/usr/bin/env bash\nset -euo pipefail\n{directives}\n{preamble}\n{root}{body}"


def _compatibility(task):
    # MC seed, step limit and patience are per-task inputs inside one serial GPU job.
    parameters = None if task.get("stage") == "deep_search" else task.get("parameters") or {}
    return json.dumps({"stage": task.get("stage"), "model_version": task.get("model_version"),
                       "config_version": task.get("config_version"),
                       "upload_operation_id": task.get("upload_operation_id"),
                       "search_group_index": task.get("search_group_index"),
                       "parent_relax_round": task.get("parent_relax_round"),
                       "parameters": parameters, "resource_profile": task.get("resource_profile")},
                      sort_keys=True, default=str)


def _assign_upload_operations(state, tasks):
    """Keep every approved allocation in a separate, persistent results scope."""
    records = list(state.get("mc_second_round_allocations") or [])
    if state.get("mc_second_round_allocation"):
        records.append(state["mc_second_round_allocation"])
    allocation_by_task = {task_id: row.get("allocation_checksum")
                          for row in records for task_id in row.get("task_ids") or []}
    allocation_by_key = {}
    structure_groups = {sid: index for index, row in enumerate(state.get("generation_history") or [], 1)
                        for sid in row.get("registered_ids") or []}
    branch_groups = {}
    relax_batches = {row['batch_id']: row for row in state.get('slurm_batches') or []
                     if row.get('calculation_group') == 'Relax-screening'}
    for task in tasks:
        group = structure_groups.get(task.get("structure_id"))
        if group and task.get("branch_id"):
            branch_groups.setdefault(task["branch_id"], set()).add(group)
    for record in state.get("action_records") or []:
        action = record.get("final_action") or {}
        preview = (action.get("parameters") or {}).get("budget_preview") or {}
        for row in preview.get("allocations") or []:
            allocation_by_key[row.get("task_key")] = preview.get("allocation_checksum")
    for task in tasks:
        if not task.get("slurm_batch_id") and not task.get("search_group_index"):
            groups = branch_groups.get(task.get("branch_id")) or set()
            group = structure_groups.get(task.get("structure_id"))
            if not group and len(groups) == 1:
                group = next(iter(groups))
            group = group or task.get("generation_cycle")
            if not group and len(state.get("generation_history") or []) <= 1:
                group = 1
            if not group:
                raise ValueError(f"任务 {task.get('task_id')} 无法唯一确定 branch 生成来源")
            task["search_group_index"] = int(group)
        if task.get('stage') == 'deep_search' and not task.get('slurm_batch_id') and not task.get('parent_relax_round'):
            matches = [row for row in tasks if row.get('stage') == 'relax_and_feature'
                       and row.get('branch_id') == task.get('branch_id')
                       and row.get('slurm_batch_id') in relax_batches]
            exact = [row for row in matches if row.get('structure_id') == task.get('structure_id')]
            parents = {int(relax_batches[row['slurm_batch_id']].get('operation_index') or 1)
                       for row in exact or matches}
            if len(parents) > 1:
                raise ValueError(f"MC任务 {task.get('task_id')} 的父 Relax 不唯一")
            if not parents and relax_batches:
                raise ValueError(f"MC任务 {task.get('task_id')} 缺少对应的 Relax 批次")
            task['parent_relax_round'] = next(iter(parents), 1)
        if task.get("slurm_batch_id") or task.get("upload_operation_id"):
            continue
        if task.get("status") != "pending":
            continue
        identity = (allocation_by_task.get(task.get("task_id"))
                    or allocation_by_key.get(task.get("task_key"))
                    or task.get("parent_decision_id"))
        if not identity:
            identity = json.dumps([task.get("stage"), task.get("model_version"),
                task.get("config_version"), task.get("generation_cycle",
                    len(state.get("generation_history") or [])),
                task.get("segment_index", 0)], sort_keys=True)
        task["upload_operation_id"] = hashlib.sha256(str(identity).encode()).hexdigest()[:12]


def _task_directories_by_id(state):
    """Index only folders named by the saved batch records/manifests."""
    indexed = {}
    for batch in state.get("slurm_batches") or []:
        batch_id = batch.get("batch_id")
        paths = list(batch.get("task_directories") or [])
        if batch.get("task_directory"):
            paths.append(batch["task_directory"])
        manifest = Path(batch["manifest_path"]) if batch.get("manifest_path") else None
        upload_root = Path(batch["upload_directory"]) if batch.get("upload_directory") else None
        if manifest is not None and manifest.is_file():
            try:
                root = manifest.parent
                for row in json.loads(manifest.read_text(encoding="utf-8")):
                    input_path = str(row.get("input_path") or "").replace("\\", "/")
                    if input_path:
                        paths.append(str(root / Path(input_path).parent))
            except (OSError, ValueError, TypeError):
                pass
        if upload_root is None and manifest is not None:
            upload_root = manifest.parent
        if upload_root is not None and upload_root.is_dir() and not paths:
            paths.extend(str(path) for path in upload_root.iterdir() if path.is_dir())
        for value in dict.fromkeys(paths):
            directory = Path(value)
            task_file = directory / "task.json"
            if not task_file.is_file():
                continue
            try:
                payload = json.loads(task_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            task_id = payload.get("task_id")
            if not task_id or (batch_id and payload.get("batch_id") not in {None, batch_id}):
                continue
            indexed.setdefault((batch_id, task_id), []).append((directory, payload))
    return indexed


def _locate_task_directory(task, indexed):
    task_id = task.get("task_id")
    batch_id = task.get("batch_id") or task.get("slurm_batch_id")
    direct_paths = []
    if task.get("input_path"):
        direct_paths.append(Path(task["input_path"]).parent)
    if task.get("task_directory"):
        direct_paths.append(Path(task["task_directory"]))
    if task.get("result_path"):
        direct_paths.append(Path(task["result_path"]).parent)
    for directory in direct_paths:
        snapshot_path = directory / "task.json"
        if not snapshot_path.is_file():
            continue
        try:
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return directory, None, "task_snapshot_invalid"
        if snapshot.get("task_id") != task_id:
            return directory, snapshot, "task_snapshot_id_mismatch"
        if task.get("task_key") and snapshot.get("task_key") != task.get("task_key"):
            return directory, snapshot, "task_snapshot_key_mismatch"
        return directory, snapshot, None
    matches = indexed.get((batch_id, task_id), [])
    if not matches:
        return None, None, None
    matching = [(directory, snapshot) for directory, snapshot in matches
                if snapshot.get("task_id") == task_id
                and (not task.get("task_key") or snapshot.get("task_key") == task.get("task_key"))]
    if len(matching) != 1:
        return None, None, "task_folder_ambiguous" if matching else "task_folder_identity_mismatch"
    return matching[0][0], matching[0][1], None


def _task_result_path(task, task_directory):
    if task_directory is not None and task.get("stage") in DFT_STAGES:
        from execution_layer.remote.result_directory_name import result_directory_name
        directory = Path(task_directory)
        name = result_directory_name(directory, task["stage"])
        if name != directory.name:
            numbered = directory.parent.parent / "results" / name / "result.json"
            if numbered.is_file():
                return numbered
    value = task.get("result_path")
    if value:
        path = Path(value)
        if path.is_file():
            return path
    if task_directory is not None:
        path = Path(task_directory) / "result.json"
        if path.is_file():
            return path
    if value and Path(value).parent.parent.name == "results":
        return Path(value)
    return Path(value) if value else None


def _expected_result_identity(task, snapshot):
    expected = deepcopy(snapshot or {})
    if expected and expected.get("task_id") != task.get("task_id"):
        return {}, "task_snapshot_id_mismatch"
    if expected and task.get("task_key") and expected.get("task_key") != task.get("task_key"):
        return {}, "task_snapshot_key_mismatch"
    for key in ("task_id", "task_key", "batch_id", "config_version", "model_version",
                "task_checksum", "protocol_version", "input_file_version"):
        saved, local = task.get(key), expected.get(key)
        if saved is not None and local is not None and saved != local:
            return {}, f"task_snapshot_{key}_mismatch"
        if saved is not None:
            expected[key] = saved
    return expected, None


def _add_collection_detail(report, task_id, reason):
    if len(report["details_preview"]) < 5:
        report["details_preview"].append({"task_id": task_id, "reason": reason})


def _compact_result_for_state(result, result_path):
    """Keep scientific summaries in state; retain large per-atom arrays in result.json."""
    compact = deepcopy(result)
    compact["calculation_result_path"] = str(Path(result_path).resolve())
    outputs = compact.get("outputs")
    if isinstance(outputs, dict):
        forces = outputs.pop("forces_ev_per_angstrom", None)
        if forces is not None:
            outputs["forces_stored_in_result_file"] = True
            outputs["forces_atom_count"] = len(forces) if isinstance(forces, list) else None
    return compact


def _write(path, payload):
    temporary = Path(f"{path}.tmp"); temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
        sort_keys=True, default=str) + "\n", encoding="utf-8"); temporary.replace(path)
